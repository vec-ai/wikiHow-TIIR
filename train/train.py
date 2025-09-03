import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
# import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from utils import load_model
from data import WikihowInterleaved

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data: Optional[str] = field(
        default=None, 
        metadata={"help": "webqa / find / mmdialog"}
    )
    doc_path: Optional[str] = field(
        default=None, 
    )
    query_path: Optional[str] = field(
        default=None, 
    )
    qrels_path: Optional[str] = field(
        default=None, 
    )
    doc_image_root: Optional[str] = field(
        default=None, 
    )
    query_image_root: Optional[List[str]] = field(
        default=None, 
    )

    text_len: int = field(
        default=1024,
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    bf16_model: bool = field(
        default=True,
    )
    is_matry: bool = field(
        default=True,
    )
    do_lora: bool = field(
        default=False,
    )
    use_lora: bool = field(
        default=False,
    )
    pooler_type: str = field(
        default="last",
        metadata={"help": "pooler_type in first, last, mean, weightedmean"}
    )
    do_norm: bool = field(
        default=True,
    )
    model_name_or_path: str = field(
        default=None,
    )
    checkpoint: str = field(
        default=None,
    )
    loss_type: str = field(
        default="MeanMrl",
    )
    image_token_num: int = field(
        default=None,
    )
    r: int = field(
        default=16,
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    gc_query_chunk_size: int = field(
        default=8,
    )
    gc_doc_chunk_size: int = field(
        default=4,
    )
    grad_cache: bool = field(
        default=True,
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)
    vl_chat_processor, model, image_processor = load_model(model_args)

    train_data = WikihowInterleaved(data_args, image_processor, vl_chat_processor, data_args.qrels_path, data_args.doc_path, data_args.query_path, data_args.doc_image_root, data_args.query_image_root, shuffle=True)
    
    from trainer import DeepSeekvlTrainer
    trainer = DeepSeekvlTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=val_data if training_args.do_eval else None,
        data_collator=train_data.Collector 
    )

     # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_data))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=valid_data)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(valid_data)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()