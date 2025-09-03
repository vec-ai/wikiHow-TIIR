# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass
from typing import Dict, List

import torch
from PIL.Image import Image
from transformers import LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin

from deepseek_vl.models.image_processing_vlm import VLMImageProcessor
from deepseek_vl.utils.conversation import get_conv_template
import random
import re

class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class VLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: list
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return [len(i) for i in self.input_ids]


@dataclass
class BatchedVLChatProcessorOutput(DictOutput):
    sft_format: List[str]
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_emb_mask: torch.BoolTensor
    num_image_tokens: torch.LongTensor

    def to(self, device, dtype=torch.bfloat16):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.images_emb_mask = self.images_emb_mask.to(device)
        self.pixel_values = self.pixel_values.to(device=device, dtype=dtype)
        self.num_image_tokens = self.num_image_tokens.to(device)
        return self


class VLChatProcessor(ProcessorMixin):
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    attributes = ["image_processor", "tokenizer"]

    system_prompt = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )

    def __init__(
        self,
        image_processor: VLMImageProcessor,
        tokenizer: LlamaTokenizerFast,
        image_tag: str = "<image_placeholder>",
        num_image_tokens: int = 576,
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        image_id = self.tokenizer.vocab.get(image_tag)
        if image_id is None:
            special_tokens = [image_tag]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Add image tag = {image_tag} to the tokenizer")

        self.image_tag = image_tag
        self.num_image_tokens = num_image_tokens
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        self.max_len = 4096
        self.image_token_num = [1,4,9,16,36,64,144,576]
        self.is_matryoshka = True

        super().__init__(
            image_processor,
            tokenizer,
            image_tag,
            num_image_tokens,
            add_special_token,
            sft_format,
            mask_prompt,
            ignore_id,
            **kwargs,
        )

    def new_chat_template(self):
        conv = get_conv_template(self.sft_format)
        conv.set_system_message(self.system_prompt)
        return conv

    def apply_sft_template_for_multi_turn_prompts(
        self,
        conversations: List[Dict[str, str]],
        sft_format: str = "deepseek",
        system_prompt: str = "",
    ):
        """
        Applies the SFT template to conversation.

        An example of conversation:
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder> is Figure 1.\n<image_placeholder> is Figure 2.\nWhich image is brighter?",
                "images": [
                    "./multi-images/attribute_comparison_1.png",
                    "./multi-images/attribute_comparison_2.png"
                ]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        Args:
            conversations (List[Dict]): A conversation with a List of Dict[str, str] text.
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """

        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        for message in conversations:
            conv.append_message(message["role"], message["content"].strip())
        sft_prompt = conv.get_prompt().strip()

        return sft_prompt

    @property
    def image_token(self):
        return self.image_tag

    @property
    def image_id(self):
        image_id = self.tokenizer.vocab.get(self.image_tag)
        return image_id

    @property
    def pad_id(self):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        return pad_id

    def get_random_image_token_num_batch(self, input_ids, image_indices):
        if self.image_token_num is not None:
            return self.image_token_num

        numbers = [1,4,9,16,36,64,144,576]
        chosen_number = random.choice(numbers)  
        return chosen_number

        
    def add_image_token(
        self,
        image_indices: List[int],
        input_ids: torch.LongTensor,
        image_token_num = None,
    ):
        """

        Args:
            image_indices (List[int]): [index_0, index_1, ..., index_j]
            input_ids (torch.LongTensor): [N]

        Returns:
            input_ids (torch.LongTensor): [N + image tokens]
            num_image_tokens (torch.IntTensor): [n_images]
        """
        if image_token_num is None:
            image_token_num = self.get_random_image_token_num_batch(input_ids,image_indices)

        input_slices = []

        start = 0
        for index in image_indices:
            if self.add_special_token:
                end = index + 1
            else:
                end = index
            input_slices.append(input_ids[start:end])
            input_slices.append(
                self.image_id * torch.ones((image_token_num,), dtype=torch.long)
            )
            start = index + 1

        input_slices.append(input_ids[start:])
        input_ids = torch.cat(input_slices, dim=0)
        num_image_tokens = torch.IntTensor([image_token_num] * len(image_indices))
        return input_ids, num_image_tokens

    def process_one(
        self,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image] = None,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert (
            prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        if prompt is None:
            sft_format = self.apply_sft_template_for_multi_turn_prompts(
                conversations=conversations,
                sft_format=self.sft_format,
                system_prompt=self.system_prompt,
            )
        else:
            sft_format = prompt
        input_ids = self.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)
        image_token_mask: torch.BoolTensor = input_ids == self.image_id
        image_indices = image_token_mask.nonzero()
        input_ids, num_image_tokens = self.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )
        images_outputs = self.image_processor(images, return_tensors="pt") if images is not None else None
        prepare = VLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=input_ids,
            pixel_values=images_outputs.pixel_values if images is not None else None,
            num_image_tokens=num_image_tokens,
        )

        return prepare


    def process_batch(
        self,
        prompt: str = None,
        conversations: List[List[Dict[str, str]]] = None,
        images: List[List[Image]] = None,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert (
            prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        prepares = []
        input_ids = []
        image_indices = []
        images_outputs = []
        sft_formats = []
        for i in range(len(conversations)):
            if prompt is None:
                sft_format = self.apply_sft_template_for_multi_turn_prompts(
                    conversations=conversations[i],
                    sft_format=self.sft_format,
                    system_prompt=self.system_prompt,
                )
            else:
                sft_format = prompt

            sft_format = re.sub(r'(<｜end▁of▁sentence｜>)+', '<｜end▁of▁sentence｜>', sft_format)
            sft_formats += [sft_format]
            input_id = self.tokenizer.encode(sft_format)
            input_ids += [torch.LongTensor(input_id)]
            image_token_mask: torch.BoolTensor = input_ids[-1] == self.image_id
            image_indices += [image_token_mask.nonzero()]
            images_outputs += [self.image_processor(images[i], return_tensors="pt") if images is not None else None]

        image_token_num = self.get_random_image_token_num_batch(input_ids,image_indices)
        for i in range(len(conversations)):
            if isinstance(image_token_num, list):
                out_ids = []
                num_image_tokens = []
                for j,itn in enumerate(image_token_num):
                    out_id, num_image_token = self.add_image_token(
                        image_indices=image_indices[i],
                        input_ids=input_ids[i],
                        image_token_num=itn
                    )
                    out_ids.append(out_id)
                    num_image_tokens.append(num_image_token)
                prepares += [VLChatProcessorOutput(
                    sft_format=sft_formats[i],
                    input_ids=out_ids,
                    pixel_values=images_outputs[i].pixel_values if images is not None else None,
                    num_image_tokens=num_image_tokens,
                )]
            else:
                out_ids, num_image_tokens = self.add_image_token(
                    image_indices=image_indices[i],
                    input_ids=input_ids[i],
                    image_token_num=image_token_num
                )
                prepares += [VLChatProcessorOutput(
                    sft_format=sft_formats[i],
                    input_ids=out_ids[i],
                    pixel_values=images_outputs[i].pixel_values if images is not None else None,
                    num_image_tokens=num_image_tokens,
                )]

        return prepares


    def __call__(
        self,
        *,
        prompt: str = None,
        conversations: List[List[Dict[str, str]]]= None,
        images: List[Image] = None,
        force_batchify: bool = True,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            force_batchify (bool): force batchify the inputs;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """
        prepares = self.process_batch(
            prompt=prompt, conversations=conversations, images=images if images is not None else None
        )

        if force_batchify:
            prepares = self.batchify_colmrl(prepares)

        return prepares

    def batchify(
        self, prepare_list: List[VLChatProcessorOutput]
    ) -> BatchedVLChatProcessorOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            prepare_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.

        Returns:
            BatchedVLChatProcessorOutput: A dictionary of the inputs to use for multimodal inference.
        """

        batch_size = len(prepare_list)
        sft_format = []
        n_images = []
        seq_lens = []

        num_image_tokens = []

        for prepare in prepare_list:
            n_images.append(len(prepare.num_image_tokens))
            seq_lens.append(len(prepare))

            if len(prepare.num_image_tokens) > 0:
                num_image_tokens.append(prepare.num_image_tokens[0])
            elif self.image_token_num is not None:
                num_image_tokens.append(self.image_token_num)
            else:
                num_image_tokens.append(576)

        num_image_tokens = torch.LongTensor(num_image_tokens).min()

        input_token_max_len = min(max(seq_lens),self.max_len)
        max_n_images = max(1, max(n_images))
        max_n_images = min(max_n_images,input_token_max_len // num_image_tokens)

        batched_input_ids = torch.full(
            (batch_size, input_token_max_len), self.pad_id
        ).long()  
        batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
        batched_pixel_values = torch.zeros(
            (batch_size, max_n_images, *self.image_processor.default_shape)
        ).float()
        batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
        batched_images_emb_mask = torch.zeros(
            (batch_size, max_n_images, num_image_tokens)
        ).bool()
        ass_token_id = self.tokenizer.convert_tokens_to_ids("Assistant")
        
        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids
            seq_len = min(len(prepare),self.max_len)

            last_ass_token = torch.nonzero(input_ids == ass_token_id)[-1]
            data_len = len(input_ids)-last_ass_token
            temp_input_ids = torch.cat((input_ids[:seq_len-data_len],input_ids[last_ass_token:]))

            image_ids_num = torch.nonzero(temp_input_ids == self.image_id).shape[0]
            image_num = image_ids_num // num_image_tokens 
            image_idx = image_ids_num % num_image_tokens
            if image_idx > 0:
                seq_len -= image_idx
                temp_input_ids = torch.cat((input_ids[:seq_len-data_len],input_ids[last_ass_token:]))
            n_image = image_num
            batched_attention_mask[i, -seq_len:] = 1
            batched_input_ids[i, -seq_len:] = torch.LongTensor(temp_input_ids)
            batched_images_seq_mask[i, -seq_len:] = temp_input_ids == self.image_id

            if n_image > 0:
                batched_pixel_values[i, :n_image] = prepare.pixel_values[:n_image]
                for j, n_image_tokens in enumerate(prepare.num_image_tokens[:n_image]):
                    batched_images_emb_mask[i, j, :n_image_tokens] = True

            sft_format.append(prepare.sft_format)

        batched_prepares = BatchedVLChatProcessorOutput(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            pixel_values=batched_pixel_values,
            images_seq_mask=batched_images_seq_mask,
            images_emb_mask=batched_images_emb_mask,
            sft_format=sft_format,
            num_image_tokens=num_image_tokens
        )

        return batched_prepares

    
    def batchify_colmrl(
        self, prepare_list: List[VLChatProcessorOutput]
    ) -> BatchedVLChatProcessorOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            prepare_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.

        Returns:
            BatchedVLChatProcessorOutput: A dictionary of the inputs to use for multimodal inference.
        """

        batch_size = len(prepare_list)
        sft_format = []
        n_images = []
        seq_lens = []

        num_image_tokens = []

        for prepare in prepare_list:
            n_images.append(len(prepare.num_image_tokens[0]))
            seq_lens += [ len(i) for i in prepare.input_ids]

        input_token_max_len = min(max(seq_lens),self.max_len)
        max_n_images = max(1, max(n_images))

        batched_input_ids = torch.full(
            (batch_size, len(self.image_token_num),input_token_max_len), self.pad_id
        ).long()  
        batched_attention_mask = torch.zeros((batch_size, len(self.image_token_num), input_token_max_len)).long()
        batched_pixel_values = torch.zeros(
            (batch_size, len(self.image_token_num), max_n_images, *self.image_processor.default_shape)
        ).float()
        batched_images_seq_mask = torch.zeros((batch_size, len(self.image_token_num), input_token_max_len)).bool()
        batched_images_emb_mask = torch.zeros(
            (batch_size, len(self.image_token_num), max_n_images, self.image_token_num[-1])
        ).bool()
        ass_token_id = self.tokenizer.convert_tokens_to_ids("Assistant")
        
        for i, prepare in enumerate(prepare_list):
            for j in range(len(self.image_token_num)):
                input_ids = prepare.input_ids[j]
                seq_len = min(len(input_ids),self.max_len)

                last_ass_token = torch.nonzero(input_ids == ass_token_id)[-1]
                data_len = len(input_ids)-last_ass_token
                temp_input_ids = torch.cat((input_ids[:seq_len-data_len],input_ids[last_ass_token:]))

                image_ids_num = torch.nonzero(temp_input_ids == self.image_id).shape[0]
                image_num = image_ids_num // self.image_token_num[j] 
                image_idx = image_ids_num % self.image_token_num[j]
                if image_idx > 0:
                    seq_len -= image_idx
                    temp_input_ids = torch.cat((input_ids[:seq_len-data_len],input_ids[last_ass_token:]))


                n_image = image_num
                batched_attention_mask[i, j, -seq_len:] = 1
                batched_input_ids[i, j, -seq_len:] = torch.LongTensor(temp_input_ids)
                batched_images_seq_mask[i, j, -seq_len:] = temp_input_ids == self.image_id

                if n_image > 0:
                    for k, n_image_tokens in enumerate(prepare.num_image_tokens[j][:n_image]):
                        batched_images_emb_mask[i, j, k, :n_image_tokens] = True
                if n_image > 0:
                    batched_pixel_values[i, j, :n_image] = prepare.pixel_values[:n_image]
                sft_format.append(prepare.sft_format)
        batched_input_ids = batched_input_ids.reshape(-1, input_token_max_len)
        batched_attention_mask = batched_attention_mask.reshape(-1,input_token_max_len)
        batched_pixel_values = batched_pixel_values.reshape(-1,max_n_images, *self.image_processor.default_shape)
        batched_images_seq_mask = batched_images_seq_mask.reshape(-1,input_token_max_len)
        batched_images_emb_mask = batched_images_emb_mask.reshape(-1,max_n_images, self.image_token_num[-1])
        nit = torch.LongTensor(self.image_token_num).expand(batch_size, -1).reshape(-1)
        batched_prepares = BatchedVLChatProcessorOutput(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            pixel_values=batched_pixel_values,
            images_seq_mask=batched_images_seq_mask,
            images_emb_mask=batched_images_emb_mask,
            sft_format=sft_format,
            num_image_tokens=nit
        )

        return batched_prepares


