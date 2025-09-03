import numpy as np
import importlib
import math
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.cuda.amp import autocast

from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from einops import rearrange
except ImportError:
    rearrange = None
from torch import nn
from deepseek_vl.models.modeling_vlm import MultiModalityPreTrainedModel,model_name_to_cls,MultiModalityConfig
from deepseek_vl.models.clip_encoder import CLIPVisionTower, HybridVisionTower
from deepseek_vl.models.projector import MlpProjector

from loss import MeanMrlDistributedContrastiveLoss,FullMrlDistributedContrastiveLoss
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import os
import copy


class DeepSeekForEmbedding(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig, pooler_type='last', do_norm=True, padding_side='left', loss_type='MeanMrl'):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

        self.pooler_type = pooler_type
        assert pooler_type in {'first', 'last', 'mean', 'weightedmean'}
        self.do_norm = do_norm
        self.padding_side = padding_side

        self.logit_scale = torch.tensor(np.log(1 / 0.05), dtype=torch.float32)
        if loss_type == 'MeanMrl':
            self.loss_fct = MeanMrlDistributedContrastiveLoss(n_hard_negatives=1)
        elif loss_type == 'mrl':
            self.loss_fct = FullMrlDistributedContrastiveLoss(n_hard_negatives=1)

        self.do_lora = False
        self.image_token_num = [1,4,9,16,36,64,144,576]
        

    def lora(self,config):
        self.language_model = get_peft_model(self.language_model, config)
        self.language_model.print_trainable_parameters()
        self.do_lora = True

    def save_pretrained(self, output_path):
        if self.do_lora:
            self.language_model.save_pretrained(output_path)
        else:
            self.language_model.save_pretrained(output_path, safe_serialization=False)
        torch.save(self.vision_model.state_dict(), os.path.join(output_path, 'vision_model.pt'))
        torch.save(self.aligner.state_dict(), os.path.join(output_path, 'aligner.pt'))


    def load_pretrained(self, output_path, use_lora=False):
        if use_lora:
            self.language_model = PeftModel.from_pretrained(self.language_model, output_path)
        else:
            self.language_model.from_pretrained(output_path)
        vision_model_states = torch.load(os.path.join(output_path, 'vision_model.pt'))
        self.vision_model.load_state_dict(vision_model_states)
        aligner_states = torch.load(os.path.join(output_path, 'aligner.pt'))
        self.aligner.load_state_dict(aligner_states)


    def matryoshka_vis_token_process(self, image_features, matryoshka_vis_token_scale):
        N, H_W, C = image_features.shape
        if H_W == matryoshka_vis_token_scale:
            return image_features
        H = W = int(H_W ** 0.5)
        reshaped_tensor = image_features.view(N, H, W, C)
        reshaped_tensor = reshaped_tensor.permute(0, 3, 1, 2)
        pool_size = stride = int( np.sqrt(H_W / matryoshka_vis_token_scale) )
        pooled_tensor = F.avg_pool2d(reshaped_tensor, kernel_size=pool_size, stride=stride)
        image_features = pooled_tensor.permute(0, 2, 3, 1)
        image_features = image_features.reshape(N, -1, C)
        return image_features

    
    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        num_image_tokens: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        if n==0:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            return inputs_embeds
        images_embeds = rearrange(pixel_values, "b n c h w -> (b n) c h w")

        images_embeds = self.vision_model(images_embeds)
        images_embeds = self.aligner(images_embeds)
        
        random_matryoshka_vis_token_scale = num_image_tokens.item()
        images_embeds = self.matryoshka_vis_token_process(images_embeds,random_matryoshka_vis_token_scale)
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        images_emb_mask = rearrange(images_emb_mask[:,:,:num_image_tokens], "b n t -> b (n t)")
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask].to(inputs_embeds.dtype)
        return inputs_embeds


    def prepare_inputs_embeds_colmrl(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        num_image_tokens: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, 8, T]
            pixel_values (torch.FloatTensor):   [b, 8, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, 8, T]
            images_emb_mask (torch.BoolTensor): [b, 8, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        if n==0:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            return inputs_embeds
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        images_embeds = self.vision_model(images)
        images_embeds = self.aligner(images_embeds)
        
        input_ids[input_ids < 0] = 0  #
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        for i,num_image_token in enumerate(self.image_token_num):
            random_matryoshka_vis_token_scale = num_image_token
            temp_images_embeds = images_embeds.clone()
            temp_images_emb_mask = images_emb_mask[:,i,:,:num_image_token].clone()
            temp_images_embeds = self.matryoshka_vis_token_process(temp_images_embeds,random_matryoshka_vis_token_scale)
            temp_images_embeds = rearrange(temp_images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
            temp_images_emb_mask = rearrange(temp_images_emb_mask, "b n t -> b (n t)")
            inputs_embeds[:,i][images_seq_mask[:,i]] = temp_images_embeds[temp_images_emb_mask].to(inputs_embeds.dtype)

        return inputs_embeds


    def get_embeds(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        sft_format: Optional[str] = None,
        pixel_values: torch.FloatTensor = None,
        images_seq_mask: torch.LongTensor = None,
        images_emb_mask: torch.LongTensor = None,
        num_image_tokens: torch.LongTensor = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids,pixel_values,images_seq_mask,images_emb_mask,num_image_tokens)

        transformer_outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            # past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embedding = self.pooling(transformer_outputs.hidden_states[-1], attention_mask)
        if self.do_norm:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def pooling(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pooler_type == 'first':
            return hidden_state[:, 0]
        else:
            assert attention_mask is not None, f"pooling {self.pooler_type} needs attention_mask"

        if attention_mask.ndim == 2:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size())
        elif attention_mask.ndim == 3:
            mask_expanded = attention_mask
        else:
            raise RuntimeError(f"Unexpected {attention_mask.ndim=}")

        hidden_state = hidden_state * mask_expanded

        if self.pooler_type == 'last':
            if self.padding_side == 'right':
                n, l, h = hidden_state.shape

                # Get shape [n] indices of the last token (i.e. the last token for each batch item)
                # Any sequence where min == 1, we use the entire sequence lenth since argmin = 0
                values, indices = torch.min(attention_mask, 1, keepdim=False)
                gather_indices = torch.where(values == 0, indices, l) - 1 # Shape [n]

                # There are empty sequences, where the index would become -1 which will crash
                gather_indices = torch.clamp(gather_indices, min=0)

                # Turn indices from shape [n] --> [n, 1, h]
                gather_indices = gather_indices.unsqueeze(1).unsqueeze(1).expand(n, 1, h)

                # Gather along the 1st dim (l) (n, l, h -> n, h)
                pooled_output = torch.gather(hidden_state, 1, gather_indices).squeeze(dim=1)
            elif self.padding_side == 'left':
                pooled_output = hidden_state[:,-1]
            else:
                raise ValueError("padding_side should be left or right")

        elif self.pooler_type == 'mean':
            lengths = mask_expanded.sum(1).clamp(min=1e-9)
            pooled_output = hidden_state.sum(dim=1) / lengths

        elif self.pooler_type == 'weightedmean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            # hidden_state shape: [bs, seq, hidden_dim]
            weights = (
                    torch.arange(start=1, end=hidden_state.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(hidden_state.size())
                    .float().to(hidden_state.device)
                )
            assert weights.shape == hidden_state.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights
            
            sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask

        else:
            raise ValueError(f"Wrong pooler : {self.pooler_type}")

        return pooled_output


    def forward(
        self,
        vl_data_query = None,
        vl_data_doc = None,
    ):
        query_embedding = None
        candidate_embeddings = None
        if vl_data_query is not None:
            query_embedding = self.get_embeds(**vl_data_query)  # Process queries
            query_embedding = F.normalize(query_embedding, dim=-1)
        if vl_data_doc is not None:
            candidate_embeddings = self.get_embeds(**vl_data_doc)  # Process queries
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
        if vl_data_query is None or vl_data_doc is None:
            return query_embedding if query_embedding is not None else candidate_embeddings
        
        logit_scale = self.logit_scale.exp()
        loss = self.loss_fct(query_embedding, candidate_embeddings, logit_scale)
        return (loss,)