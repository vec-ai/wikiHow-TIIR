# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from itertools import repeat
from typing import Any, Dict, Optional, Union, List, Tuple, Mapping, Callable
import torch
from torch import nn, Tensor
from torch.utils.data import ConcatDataset, DataLoader, IterableDataset

from transformers.trainer import Trainer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import seed_worker, EvalLoopOutput, has_length, EvalPrediction, denumpify_detensorize
from transformers.training_args import ParallelMode
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
import numpy as np
from my_grad_cache import GradCache
from torch.cuda.amp import GradScaler, autocast

def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()

def smp_forward_only(model, inputs):
    return model(**inputs)

def smp_nested_concat(tensor):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(smp_nested_concat(t) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: smp_nested_concat(v) for k, v in tensor.items()})
    return tensor.concat().detach().cpu()

logger = logging.getLogger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

class DeepSeekvlTrainer(Trainer):
    """
        model: DeepSeekvl,
        args:DeepSeekvlTrainingArguments,
        data_collator: DeepSeekvlCollator,
        train_dataset: Optional[Dataset] = None,
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.args.grad_cache:
            logger.warning("Grad cache should be the last method to enlarge the batch_size...")
            if self.is_deepspeed_enabled or is_sagemaker_mp_enabled() or self.use_apex:
                raise ValueError(
                    "Using grad_cache together with deepspeed/sagemaker_mp/apex is not possible."
                )

            scaler = self.scaler if getattr(self, 'do_grad_scaling', False) else self.accelerator.scaler
            self.gc = GradCache(
                models=[self.model, self.model],
                chunk_sizes=[self.args.gc_query_chunk_size, self.args.gc_doc_chunk_size],
                loss_fn=self.model.loss_fct,
                split_input_fn=split_inputs,
                get_rep_fn=get_rep,
                fp16=self.args.fp16,
                scaler=scaler if self.args.fp16 else None
            )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        logger.info(f"Loading model from {resume_from_checkpoint}.")

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, IterableDataset):
            if (isinstance(train_dataset, ConcatDataset)
                and len(train_dataset.datasets) > 1
            ):
                from sampler import MultiSubsetBatchSampler

                batch_size = dataloader_params.pop('batch_size') * self.args.world_size
                batch_sampler = MultiSubsetBatchSampler(
                    train_dataset.datasets, batch_size
                )
                dataloader_params["batch_sampler"] = batch_sampler
                logger.info(
                    f"debias_batch is True, will sample by {batch_size=} and"
                    f" split to {self.args.world_size} process(es)."
                )
            else:
                dataloader_params["sampler"] = self._get_train_sampler()
                dataloader_params["drop_last"] = True
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        logs['step'] = self.state.global_step
        self.state.log_history.append(dict(logs))
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        if self.args.grad_cache:
            queries, keys,  = {'vl_data_query': inputs['vl_data_query']}, {'vl_data_doc': inputs['vl_data_doc']}
            # _distributed = self.args.local_rank > -1
            _distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
            self.gc.models = [model, model]
            loss = self.gc(queries, keys, no_sync_except_last=_distributed,logit_scale=model.module.logit_scale.exp())
        else:
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
        
        return loss.detach() / self.args.gradient_accumulation_steps


def get_rep(x):
    return x


def split_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]
    batch_size = (len(arg_val[list(arg_val.keys())[0]])+chunk_size-1) // chunk_size
    keys = [k for k in arg_val.keys() if not isinstance(arg_val[k], list)]
    chunked_tensors = [
        arg_val[k].split(chunk_size, dim=0) if arg_val[k].dim() > 0 else [arg_val[k]]*batch_size
        for k in keys
    ]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]

