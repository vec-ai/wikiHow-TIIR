from multi_model_matry import  DeepSeekForEmbedding as DeepSeekForEmbedding_matry
import sys
import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoConfig
from deepseek_processing_vlm import VLChatProcessor
from peft import get_peft_model, LoraConfig, TaskType
import torch 

def load_model(args):
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_name_or_path)

    if args.bf16_model:
        model = DeepSeekForEmbedding_matry.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16)
    else:
        model = DeepSeekForEmbedding_matry.from_pretrained(args.model_name_or_path)
    vl_chat_processor.is_matryoshka = True

    if args.checkpoint is not None:
        model.load_pretrained(args.checkpoint,args.use_lora)
    if args.do_lora:
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=args.r, lora_alpha=32, lora_dropout=0.1,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'up_proj', 'gate_proj']
        )
        model.lora(peft_config)
    if args.image_token_num is not None:
        vl_chat_processor.image_token_num = args.image_token_num
    return vl_chat_processor, model, None

