import torch
import torch.nn.functional as F
# from peft import PeftModel
from transformers import AutoTokenizer, AutoModel


import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from transformers import AutoModel, AutoConfig
from transformers import LlavaNextProcessor
from transformers import LlavaNextForConditionalGeneration, LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast, image_size_to_num_patches

class NVMMEmbedModel(LlavaNextForConditionalGeneration):
    def __init__(self, config: LlavaNextConfig):
        super().__init__(config)

        nvemb_config = AutoConfig.from_pretrained(config.retriever, trust_remote_code=True)
        nvemb_model = AutoModel.from_config(nvemb_config, trust_remote_code=True)
        self.language_model = nvemb_model.embedding_model
        self.latent_attention_model = nvemb_model.latent_attention_model

        self.preprocess_fn = LlavaNextProcessor.from_pretrained(config._name_or_path)
        self.preprocess_fn.tokenizer.padding_side = config.padding_side
        self.preprocess_fn.tokenizer.add_eos_token = config.add_eos_token
        self.global_image_patch_only = config.global_image_patch_only


    def create_pool_mask(self, attention_mask, instruction_lengths):
        pool_mask = attention_mask.clone()
        if instruction_lengths.unique().shape[0] == 1:
            length = instruction_lengths[0].item()
            pool_mask[:, :length] = 0
        else:
            for i, length in enumerate(instruction_lengths): 
                pool_mask[i, :length] = 0
        return pool_mask

    def calculate_instruction_length(self, tokenizer, prompts, prefix):
        instructions = []
        instruction_lengths = []
        for prompt in prompts:
            if prefix in prompt:
                instruction = prompt.split(prefix)[0]
                input_ids = tokenizer(instruction, return_tensors=None)['input_ids']
                instruction_length = len(input_ids)
                if '<image>' in instruction:
                    instruction_length += (576 - 1)
                instruction_lengths.append(instruction_length)
            else:
                instruction_lengths.append(0)
        return instruction_lengths

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        instruction_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        clip_global_image_feature = None

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
            for_inputs_embeds_ids = input_ids.clone()
            for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0
            for_inputs_embeds_ids[(input_ids == 32001)] = 2 #We use tokenizer from Llava-Next but later replace PAD with EOS Token
            inputs_embeds = self.language_model.get_input_embeddings()(for_inputs_embeds_ids)
            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
                # ! infer image_num_patches from image_sizes
                image_num_patches = [
                    image_size_to_num_patches(
                        image_size=imsize,
                        grid_pinpoints=self.config.image_grid_pinpoints,
                        patch_size=self.config.vision_config.image_size,
                    )
                    for imsize in image_sizes
                ]
                # figure out if pixel_values is concatenated or stacked
                if pixel_values.dim() == 5:
                    # stacking when input is (batch_size, num_patches, num_channels, height, width)
                    _pixel_values_list = [
                        pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
                    ]
                    if pixel_values.shape[1] == 1:
                        image_num_patches = [1 for imsize in image_sizes]
                    pixel_values = torch.cat(_pixel_values_list, dim=0)
                elif pixel_values.dim() != 4:
                    # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                    raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

                image_features = self.vision_tower(pixel_values, output_hidden_states=True)
                clip_global_image_feature = image_features.pooler_output
                selected_image_feature = image_features.hidden_states[vision_feature_layer]
                
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                
                image_features = self.multi_modal_projector(selected_image_feature)
                image_features = torch.split(image_features, image_num_patches, dim=0)

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"

                image_features, feature_lens = self.pack_image_features(
                    image_features,
                    image_sizes,
                    image_newline=self.image_newline,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )

                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, position_ids, labels, _ = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )

            # pixel_values is not None but is empty ---> text only cases
            elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
                # there are no images
                pass

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)

                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pool_mask = self.create_pool_mask(attention_mask, instruction_lengths)
        
        embeds = self.latent_attention_model(
                outputs.last_hidden_state,
                pool_mask,
        )


        return LlavaNextCausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=embeds,
            attentions=outputs.attentions,
            image_hidden_states=clip_global_image_feature,
        )

    @torch.no_grad()
    def encode(self, inputs, is_query = False, instruction = None, max_length = 512, query_prefix = 'Query: '):
        assert type(inputs) == list, 'inputs should be a list of dictionay'
        prompts, imgs = [], []
        if is_query:
            if instruction is not None:
                prompt_template = f"Instruct: {instruction}\n{query_prefix}<image>\n<text>"
            else:
                prompt_template = f"{query_prefix}<image>\n<text>"
        else:
            prompt_template = f"<image>\n<text>"
    
        for input_ in inputs:
            if 'img' in input_:
                imgs.append(input_['img'])
                prompt = prompt_template
            else:
                prompt = prompt_template.replace('<image>\n', '')

            if ('txt' in input_) and (input_['txt'] is not None):
                prompt = prompt.replace('<text>', input_['txt'])
            else:
                prompt = prompt.replace('<text>', '')
            
            prompts.append(prompt)
        
        if len(imgs) == 0:
            imgs = None
        collated_features = self.preprocess_fn(prompts, imgs, return_tensors="pt", padding="longest", max_length=max_length, truncation=True).to(self.device)
        if self.global_image_patch_only and (imgs is not None): # we only use global image patch as default
            collated_features['pixel_values'] = collated_features['pixel_values'][:, 0:1]

        instruction_lengths = self.calculate_instruction_length(self.preprocess_fn.tokenizer, prompts, f'\n{query_prefix}')
        collated_features['instruction_lengths'] = torch.tensor(instruction_lengths).to(self.device)

        return self(**collated_features)
    

    @torch.no_grad()
    def prepare(self, inputs, is_query = False, instruction = None, max_length = 512, query_prefix = 'Query: '):
        assert type(inputs) == list, 'inputs should be a list of dictionay'
        prompts, imgs = [], []
        if is_query:
            if instruction is not None:
                prompt_template = f"Instruct: {instruction}\n{query_prefix}<image>\n<text>"
            else:
                prompt_template = f"{query_prefix}<image>\n<text>"
        else:
            prompt_template = f"<image>\n<text>"
    
        for input_ in inputs:
            if 'img' in input_:
                imgs.append(input_['img'])
                prompt = prompt_template
            else:
                prompt = prompt_template.replace('<image>\n', '')

            if ('txt' in input_) and (input_['txt'] is not None):
                prompt = prompt.replace('<text>', input_['txt'])
            else:
                prompt = prompt.replace('<text>', '')
            
            prompts.append(prompt)
        
        if len(imgs) == 0:
            imgs = None
        collated_features = self.preprocess_fn(prompts, imgs, return_tensors="pt", padding="longest", max_length=max_length, truncation=True)
        if self.global_image_patch_only and (imgs is not None): # we only use global image patch as default
            collated_features['pixel_values'] = collated_features['pixel_values'][:, 0:1]

        instruction_lengths = self.calculate_instruction_length(self.preprocess_fn.tokenizer, prompts, f'\n{query_prefix}')
        collated_features['instruction_lengths'] = torch.tensor(instruction_lengths)

        return collated_features



AutoModel.register(LlavaNextConfig, NVMMEmbedModel)
NVMMEmbedModel.register_for_auto_class("AutoModel")