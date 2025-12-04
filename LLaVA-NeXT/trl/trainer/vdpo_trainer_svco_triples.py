# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch.nn import Module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from ..import_utils import is_peft_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)
from ..diff_lib import get_diff_ids



if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

def chip_get_batch_logps(logits: torch.FloatTensor,
                         reference_logits: torch.FloatTensor,
                         uncond_ref_logits: torch.FloatTensor,
                         labels: torch.LongTensor,
                         average_log_prob: bool = False):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        uncond_ref_logits: Logits of the reference model (unconditional unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """

    labels = labels[:-1, :].clone()
    logits = logits[:-1, :, :]  # 因为最后一个是diffsuion那部分的logits，所以要去掉，只取yw，yl那部分的logits
    assert logits.shape[:-1] == labels.shape, (logits.shape[:-1], labels.shape)
    assert reference_logits.shape[:-1] == labels.shape, (reference_logits.shape[:-1], labels.shape)
    assert uncond_ref_logits.shape[:-1] == labels.shape, (uncond_ref_logits.shape[:-1], labels.shape)

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    uncond_ref_logits = uncond_ref_logits[:, :-1, :]

    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    uncond_ref_vocab_logps = uncond_ref_logits.log_softmax(-1)

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_policy_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_uncond_ref_token_logps = torch.gather(uncond_ref_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    # 以上四个公式是公式（8）~（9），可不要

    if average_log_prob:
        return (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_policy_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_reference_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_uncond_ref_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
               per_policy_token_logps, per_reference_token_logps, per_uncond_ref_token_logps
    else:
        return (per_position_kl * loss_mask).sum(-1), \
               (per_policy_token_logps * loss_mask).sum(-1), \
               (per_reference_token_logps * loss_mask).sum(-1), \
               (per_uncond_ref_token_logps * loss_mask).sum(-1), \
               per_policy_token_logps, per_reference_token_logps, per_uncond_ref_token_logps

def chip_get_batch_logps_symmetrical(logits: torch.FloatTensor,
                         no_image_logits: torch.FloatTensor,
                         reference_logits: torch.FloatTensor,
                         uncond_ref_logits: torch.FloatTensor,
                         no_image_ref_logits: torch.FloatTensor,
                         labels: torch.LongTensor,
                         labels_noimage: torch.LongTensor,
                         average_log_prob: bool = False):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        uncond_ref_logits: Logits of the reference model (unconditional unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """
    labels = labels[:-2, :].clone()
    labels_noimage = labels_noimage.clone()
    logits = logits[:-2, :, :]  # 因为最后一个是diffsuion那部分的logits，所以要去掉，只取yw，yl那部分的logits
    assert logits.shape[:-1] == labels.shape, (logits.shape[:-2], labels.shape)
    assert reference_logits.shape[:-1] == labels.shape, (reference_logits.shape[:-2], labels.shape)
    assert uncond_ref_logits.shape[:-1] == labels.shape, (uncond_ref_logits.shape[:-2], labels.shape)

    labels = labels[:, 1:].clone()
    labels_noimage = labels_noimage[:, 1:].clone()
    logits = logits[:, :-1, :]
    no_image_logits = no_image_logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    uncond_ref_logits = uncond_ref_logits[:, :-1, :]
    no_image_ref_logits = no_image_ref_logits[:, :-1, :]

    loss_mask = (labels != -100)
    loss_mask_noimage = (labels_noimage != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    labels_noimage[labels_noimage == -100] = 0

    vocab_logps = logits.log_softmax(-1)
    no_image_vocab_logps = no_image_logits.log_softmax(-1)
    uncond_ref_vocab_logps = uncond_ref_logits.log_softmax(-1)
    no_image_ref_vocab_logps = no_image_ref_logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_policy_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_policy_no_image_policy_token_logps = torch.gather(no_image_vocab_logps, dim=2, index=labels_noimage.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_uncond_ref_token_logps = torch.gather(uncond_ref_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_no_image_ref_token_logps = torch.gather(no_image_ref_vocab_logps, dim=2, index=labels_noimage.unsqueeze(2)).squeeze(2)
    # 以上四个公式是公式（8）~（9），可不要

    if average_log_prob:
        return (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_policy_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_policy_no_image_policy_token_logps * loss_mask_noimage).sum(-1) / loss_mask_noimage.sum(-1), \
               (per_reference_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_uncond_ref_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_no_image_ref_token_logps * loss_mask_noimage).sum(-1) / loss_mask_noimage.sum(-1), \
               per_policy_token_logps, per_reference_token_logps, per_uncond_ref_token_logps, per_policy_no_image_policy_token_logps, per_no_image_ref_token_logps
    else:
        return (per_position_kl * loss_mask).sum(-1), \
               (per_policy_token_logps * loss_mask).sum(-1), \
               (per_policy_no_image_policy_token_logps * loss_mask_noimage).sum(-1), \
               (per_reference_token_logps * loss_mask).sum(-1), \
               (per_uncond_ref_token_logps * loss_mask).sum(-1), \
               (per_no_image_ref_token_logps * loss_mask_noimage).sum(-1), \
               per_policy_token_logps, per_reference_token_logps, per_uncond_ref_token_logps, per_policy_no_image_policy_token_logps, per_no_image_ref_token_logps



def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False,
                    return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_policy_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                          index=labels.unsqueeze(2)).squeeze(2)

    policy_vocab_ps = logits.softmax(-1)

    log_prob = (per_policy_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    if return_per_token_logp:
        return per_policy_token_logps

    if return_all:
        return per_policy_token_logps, log_prob, average_log_prob, policy_vocab_ps, loss_mask

    return log_prob, average_log_prob

class SVCOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper.
        label_smoothing (`float`, defaults to 0):
            The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report that should be between 0 and 0.5.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either `"sigmoid"` the default DPO loss,`"hinge"` loss from [SLiC](https://arxiv.org/abs/2305.10425) paper, `"ipo"` from [IPO](https://arxiv.org/abs/2310.12036) paper, or `"kto"` from the HALOs [report](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf).
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Flag to precompute reference model log probabilities and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        ref_model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string
        model_adapter_name (`str`, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        reference_free (`bool`):
            If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.
    """

    _tag_names = ["trl", "dpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        dpo_alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid",
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
    ):
        # import pdb;pdb.set_trace()
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError("You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated.")

        if isinstance(model, str):
            warnings.warn("You passed a model_id to the DPOTrainer. This will automatically create an " "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you.")
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn("You passed a ref model_id to the DPOTrainer. This will automatically create an " "`AutoModelForCausalLM`")
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if generate_during_eval and not is_wandb_available():
            raise ValueError("`generate_during_eval=True` requires Weights and Biases to be installed." " Please install `wandb` to resolve.")

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name
        self.reference_free = reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            if is_deepspeed_zero3_enabled():
                self.ref_model = AutoModelForCausalLM.from_pretrained(model)
            else:
                self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOTrainer's init" " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOTrainer's init" " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init" " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments" " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn("You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter.")

        self.dpo_alpha = dpo_alpha
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.dataset_num_proc = dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        # with PartialState().local_main_process_first():
        #     # tokenize the dataset
        #     train_dataset = train_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)
        #     if eval_dataset is not None:
        #         eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError("Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`.")

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError("You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`.")

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError("No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`")
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = max(model.config.hidden_sizes) if getattr(model.config, "hidden_sizes", None) else getattr(model.config, "hidden_size", None)
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics((reference_chosen_logp, reference_rejected_logp))
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(name="reference_chosen_logps", column=all_reference_chosen_logps)
            self.train_dataset = self.train_dataset.add_column(name="reference_rejected_logps", column=all_reference_rejected_logps)

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics((reference_chosen_logp, reference_rejected_logp))
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="reference_chosen_logps", column=all_reference_chosen_logps)
            eval_dataset = eval_dataset.add_column(name="reference_rejected_logps", column=all_reference_rejected_logps)

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum([a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])])
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError("Chosen and rejected prompt_input_ids might only differ on the " "last token due to tokenizer merge ops.")

            # add BOS token to head of prompt
            prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
            chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
            rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]}
            rejected_sequence_tokens = {k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]}
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [self.label_pad_token_id] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [self.label_pad_token_id] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True)
            rejected_tokens = self.tokenizer(rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True)
            prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True)

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(labels=batch["rejected_labels"])
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(labels=batch["chosen_labels"])

        return batch

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(self.model).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            # import pdb; pdb.set_trace()
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
        # import pdb; pdb.set_trace()
        # repeated_list = [
        #     batch['images'][0] * 2,
        #     batch['images'][1] * 2
        # ]
        concatenated_batch["concatenated_images_3"] = batch["images"] + batch["images"] + batch["rejected_image"]
        concatenated_batch["image_sizes"] = batch["image_sizes"] + batch["image_sizes"] + batch["rejected_image_sizes"]
        concatenated_batch["modalities"] = batch["modalities"] + batch["modalities"] + batch["rejected_modalities"]
        return concatenated_batch

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios
        # print(f"pi log ratios: {pi_logratios}")
        # print(f"ref log ratios: {ref_logratios}")
        # print(f"logits: {logits}")
        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']")

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device)).detach()

        return losses, chosen_rewards, rejected_rewards

    def get_sft_loss(self, logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        # Enable model/pipeline parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test.
        CHANGE: 1. add sft loss
        2. all gather metrics
        """
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
            rejected_labels,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                        ) = self.concatenated_forward(
                            self.model, batch
                        )[:2]
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                    ) = self.concatenated_forward(
                        self.ref_model, batch
                    )[:2]

        unscaled_dpo_losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        unscaled_dpo_losses = unscaled_dpo_losses.mean()
        dpo_losses = unscaled_dpo_losses * self.dpo_alpha
        unscaled_sft_loss = self.get_sft_loss(policy_chosen_logits, chosen_labels)
        sft_loss = unscaled_sft_loss * self.gamma

        # print(sft_loss.shape, dpo_losses.shape)
        losses = dpo_losses + sft_loss
        # losses = sft_loss # sft only
        # losses = dpo_losses # dpo only
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        def all_gather_tensor(tensor):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tensor = tensor.detach()
                gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered_tensor, tensor)
                tensor = torch.cat(gathered_tensor, dim=0)
            # else:
            #     print('not distributed')
            return tensor

        # gather chosen_rewards across devices
        chosen_rewards = all_gather_tensor(chosen_rewards)
        rejected_rewards = all_gather_tensor(rejected_rewards)
        reward_accuracies = all_gather_tensor(reward_accuracies)
        policy_chosen_logps = all_gather_tensor(policy_chosen_logps)
        policy_rejected_logps = all_gather_tensor(policy_rejected_logps)
        reference_chosen_logps = all_gather_tensor(reference_chosen_logps)
        reference_rejected_logps = all_gather_tensor(reference_rejected_logps)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}losses/dpo"] = unscaled_dpo_losses.cpu()
        metrics[f"{prefix}losses/sft"] = unscaled_sft_loss.cpu()
        metrics[f"{prefix}losses/total"] = losses.cpu()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        # policy logps
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        # policy logits (exclude image tokens)
        # metrics[f"{prefix}logits/rejected"] =policy_rejected_logits
        # metrics[f"{prefix}logits/chosen"] = policy_chosen_logits
        # reference logps
        metrics[f"{prefix}ref_logps/rejected"] = reference_rejected_logps.mean().cpu()
        metrics[f"{prefix}ref_logps/chosen"] = reference_chosen_logps.mean().cpu()

        # metrics all pick .4 digits
        # for k in metrics:
        #     metrics[k] = round(metrics[k].item(), 4)

        return losses, metrics

    def chip_loss_symmetrical(self,policy_logps_img_chosen_res_chosen: torch.FloatTensor,
                    policy_logps_img_chosen_res_rejected: torch.FloatTensor,
                    policy_logps_img_rejected_res_chosen: torch.FloatTensor,
                    policy_logps_img_rejected_res_rejected: torch.FloatTensor,
                    policy_logps_no_img_res_chosen: torch.FloatTensor,
                    policy_logps_no_img_res_rejected: torch.FloatTensor,
                    reference_logps_img_chosen_res_chosen: torch.FloatTensor,
                    reference_logps_img_chosen_res_rejected: torch.FloatTensor,
                    reference_logps_img_rejected_res_chosen: torch.FloatTensor,
                    reference_logps_img_rejected_res_rejected: torch.FloatTensor,
                    reference_logps_no_img_res_chosen: torch.FloatTensor,
                    reference_logps_no_img_res_rejected: torch.FloatTensor,
                    chosen_position_kl: torch.FloatTensor,
                    rejected_position_kl: torch.FloatTensor,
                    policy_chosen_logp_v: torch.FloatTensor,
                    reference_chosen_logp_v: torch.FloatTensor,
                    causal_flag: bool,
                    diffusion_idx: bool,
                    beta: float = 0.1, gama: float = 0.3, ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the TDPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logp: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logp: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            policy_win_diffusionImage_logp: Log probabilities of the policy model for the chosen responses with diffusion image. Shape: (batch_size,)
            reference_chosen_logp: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logp: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            uncond_ref_win_logp: unconditional Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            uncond_ref_rej_logp: unconditional Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss
            gama: Temperature parameter for the CMDPO loss

        Returns:
            A tuple of two tensors: (losses, rewards).
            The losses tensor contains the TDPO loss for each example in the batch.
            The rewards tensors contain the rewards for response pair.
        """
        pi_logratios = policy_logps_img_chosen_res_chosen - policy_logps_img_chosen_res_rejected
        ref_logratios = reference_logps_img_chosen_res_chosen - reference_logps_img_chosen_res_rejected
        logits = pi_logratios - ref_logratios

        pi_logratios_rej = policy_logps_img_rejected_res_rejected - policy_logps_img_rejected_res_chosen
        ref_logratios_rej = reference_logps_img_rejected_res_rejected - reference_logps_img_rejected_res_chosen
        logits_rej = pi_logratios_rej - ref_logratios_rej
        # print("这是TDPO的loss:", -F.logsigmoid(beta * logits))
        if causal_flag:
            #if diffusion_idx:
                #import pdb; pdb.set_trace()
            # causal_logits_win_rej = (policy_chosen_logp_v - reference_chosen_logp_v) \
            #                 - (policy_win_diffusionImage_logp - uncond_ref_win_logp)
            # causal_logits_rej_win = (policy_rej_diffusionImage_logp - uncond_ref_rej_logp) \
            #             - (policy_rejected_logp - reference_rejected_logp)
            pi_img_win_vs_no_img = policy_logps_img_chosen_res_chosen - policy_logps_no_img_res_chosen
            ref_img_win_vs_no_img = reference_logps_img_chosen_res_chosen - reference_logps_no_img_res_chosen
            logits_img_win_vs_no_img = pi_img_win_vs_no_img - ref_img_win_vs_no_img

            pi_no_img_vs_img_lose = policy_logps_no_img_res_chosen - policy_logps_img_rejected_res_chosen
            ref_no_img_vs_img_lose = reference_logps_no_img_res_chosen - reference_logps_img_rejected_res_chosen
            logits_no_img_vs_img_lose = pi_no_img_vs_img_lose - ref_no_img_vs_img_lose

            pi_img_lose_vs_no_img = policy_logps_img_rejected_res_rejected - policy_logps_no_img_res_rejected
            ref_img_lose_vs_no_img = reference_logps_img_rejected_res_rejected - reference_logps_no_img_res_rejected
            logits_img_lose_vs_no_img = pi_img_lose_vs_no_img - ref_img_lose_vs_no_img

            pi_no_img_vs_img_win = policy_logps_no_img_res_rejected - policy_logps_img_chosen_res_rejected
            ref_no_img_vs_img_win = reference_logps_no_img_res_rejected - reference_logps_img_chosen_res_rejected
            logits_no_img_vs_img_win = pi_no_img_vs_img_win - ref_no_img_vs_img_win

            pi_img_win_vs_img_lose = policy_logps_img_chosen_res_chosen - policy_logps_img_rejected_res_chosen
            ref_img_win_vs_img_lose = reference_logps_img_chosen_res_chosen - reference_logps_img_rejected_res_chosen
            logits_img_win_vs_img_lose = pi_img_win_vs_img_lose - ref_img_win_vs_img_lose

            pi_img_lose_vs_img_win = policy_logps_img_rejected_res_rejected - policy_logps_img_chosen_res_rejected
            ref_img_lose_vs_img_win = reference_logps_img_rejected_res_rejected - reference_logps_img_chosen_res_rejected
            logits_img_lose_vs_img_win = pi_img_lose_vs_img_win - ref_img_lose_vs_img_win

            img_win_vs_no_img_loss = -F.logsigmoid(beta * logits_img_win_vs_no_img)
            no_img_vs_img_lose_loss = -F.logsigmoid(beta * logits_no_img_vs_img_lose)
            img_lose_vs_no_img_loss = -F.logsigmoid(beta * logits_img_lose_vs_no_img)
            no_img_vs_img_win_loss = -F.logsigmoid(beta * logits_no_img_vs_img_win)

            img_win_vs_img_lose_loss = -F.logsigmoid(beta * logits_img_win_vs_img_lose)
            img_lose_vs_img_win_loss = -F.logsigmoid(beta * logits_img_lose_vs_img_win)


            if self.args.use_symmetrical_loss:
                #logits = logits + self.args.vdpo_weight * causal_logits
                v_losses = (
                img_win_vs_no_img_loss
                + no_img_vs_img_lose_loss
                + img_lose_vs_no_img_loss
                + no_img_vs_img_win_loss
                # + img_win_vs_img_lose_loss
                # + img_lose_vs_img_win_loss
            )
            else:
                v_losses = (
                img_win_vs_no_img_loss
                + no_img_vs_img_lose_loss
                + img_lose_vs_no_img_loss
                + no_img_vs_img_win_loss
                ).detach()

            # 计算 KL 散度
            q_abs_chosen = -F.logsigmoid(beta * (policy_logps_img_chosen_res_chosen - reference_logps_img_chosen_res_chosen) - 0)
            q_abs_chosen_v = -F.logsigmoid(beta * (policy_chosen_logp_v - reference_chosen_logp_v) - 0)
            q_abs_rejected = -F.logsigmoid(-(beta * (policy_logps_img_chosen_res_rejected - reference_logps_img_chosen_res_rejected) - 0))
            q_abs_rejected_v = -F.logsigmoid(-(beta * (policy_logps_img_rejected_res_chosen - reference_logps_img_rejected_res_chosen) - 0))
            # q_abs_losses = (q_abs_chosen + q_abs_rejected)
            v_chosen_values = policy_chosen_logp_v - reference_chosen_logp_v
            v_chosen_rewards = beta * v_chosen_values.detach()
            v_rejected_values = policy_logps_img_rejected_res_chosen - reference_logps_img_rejected_res_chosen
            v_rejected_rewards = beta * v_rejected_values.detach()
        else:
            v_losses = torch.tensor(0.0, device=logits.device).detach()
            q_abs_chosen = -F.logsigmoid(beta * (policy_logps_img_chosen_res_chosen - reference_logps_img_chosen_res_chosen) - 0)
            q_abs_chosen_v = -F.logsigmoid(beta * (policy_chosen_logp_v - reference_chosen_logp_v) - 0)
            q_abs_rejected = -F.logsigmoid(-(beta * (policy_logps_img_chosen_res_rejected - reference_logps_img_chosen_res_rejected) - 0))
            q_abs_rejected_v = -F.logsigmoid(-(beta * (policy_logps_img_rejected_res_chosen - reference_logps_img_rejected_res_chosen) - 0))
            v_chosen_values = policy_chosen_logp_v - reference_chosen_logp_v
            v_chosen_rewards = beta * v_chosen_values.detach()
            v_rejected_values = policy_logps_img_rejected_res_chosen - reference_logps_img_rejected_res_chosen
            v_rejected_rewards = beta * v_rejected_values.detach()

        chosen_values = policy_logps_img_chosen_res_chosen - reference_logps_img_chosen_res_chosen
        rejected_values = policy_logps_img_chosen_res_rejected - reference_logps_img_chosen_res_rejected

        t_losses = -F.logsigmoid(beta * logits) + (-F.logsigmoid(beta * logits_rej))

        chosen_rewards = beta * chosen_values.detach()
        rejected_rewards = beta * rejected_values.detach()

        return t_losses, chosen_rewards, rejected_rewards, v_losses, q_abs_chosen, q_abs_chosen_v, q_abs_rejected, q_abs_rejected_v, v_chosen_rewards, v_rejected_rewards
    
    def chip_loss(self, policy_chosen_logp: torch.FloatTensor,
                  policy_rejected_logp: torch.FloatTensor,
                  policy_win_diffusionImage_logp: torch.FloatTensor,
                  reference_chosen_logp: torch.FloatTensor,
                  reference_rejected_logp: torch.FloatTensor,
                  uncond_ref_win_logp: torch.FloatTensor,
                  uncond_ref_rej_logp: torch.FloatTensor,
                  chosen_position_kl: torch.FloatTensor,
                  rejected_position_kl: torch.FloatTensor,
                  policy_chosen_logp_v: torch.FloatTensor,
                  reference_chosen_logp_v: torch.FloatTensor,
                  causal_flag: bool,
                  diffusion_idx: bool,
                  beta: float = 0.1, gama: float = 0.3, ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the TDPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logp: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logp: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            policy_win_diffusionImage_logp: Log probabilities of the policy model for the chosen responses with diffusion image. Shape: (batch_size,)
            reference_chosen_logp: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logp: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            uncond_ref_win_logp: unconditional Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            uncond_ref_rej_logp: unconditional Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss
            gama: Temperature parameter for the CMDPO loss

        Returns:
            A tuple of two tensors: (losses, rewards).
            The losses tensor contains the TDPO loss for each example in the batch.
            The rewards tensors contain the rewards for response pair.
        """
        pi_logratios = policy_chosen_logp - policy_rejected_logp
        ref_logratios = reference_chosen_logp - reference_rejected_logp
        logits = pi_logratios - ref_logratios
        # print("这是TDPO的loss:", -F.logsigmoid(beta * logits))
        if causal_flag:
            #if diffusion_idx:
                #import pdb; pdb.set_trace()
            causal_logits = (policy_chosen_logp_v - reference_chosen_logp_v) \
                            - (policy_win_diffusionImage_logp - uncond_ref_win_logp)
            if self.args.use_cross_modal_loss:
                #logits = logits + self.args.vdpo_weight * causal_logits
                v_losses = -F.logsigmoid(beta * causal_logits)     
            else:
                v_losses = -F.logsigmoid(beta * causal_logits).detach()

            # 计算 KL 散度
            q_abs_chosen = -F.logsigmoid(beta * (policy_chosen_logp - reference_chosen_logp) - 0)
            q_abs_chosen_v = -F.logsigmoid(beta * (policy_chosen_logp_v - reference_chosen_logp_v) - 0)
            q_abs_rejected = -F.logsigmoid(-(beta * (policy_rejected_logp - reference_rejected_logp) - 0))
            q_abs_rejected_v = -F.logsigmoid(-(beta * (policy_win_diffusionImage_logp - uncond_ref_win_logp) - 0))
            # q_abs_losses = (q_abs_chosen + q_abs_rejected)
            v_chosen_values = policy_chosen_logp_v - reference_chosen_logp_v
            v_chosen_rewards = beta * v_chosen_values.detach()
            v_rejected_values = policy_win_diffusionImage_logp - uncond_ref_win_logp
            v_rejected_rewards = beta * v_rejected_values.detach()
        else:
            v_losses = torch.tensor(0.0, device=logits.device).detach()
            q_abs_chosen = -F.logsigmoid(beta * (policy_chosen_logp - reference_chosen_logp) - 0)
            q_abs_chosen_v = -F.logsigmoid(beta * (policy_chosen_logp_v - reference_chosen_logp_v) - 0)
            q_abs_rejected = -F.logsigmoid(-(beta * (policy_rejected_logp - reference_rejected_logp) - 0))
            q_abs_rejected_v = -F.logsigmoid(-(beta * (policy_win_diffusionImage_logp - uncond_ref_win_logp) - 0))
            v_rejected_values = policy_win_diffusionImage_logp - uncond_ref_win_logp
            v_rejected_rewards = torch.exp(-beta * torch.abs(v_rejected_values)).detach()
            v_chosen_values = policy_chosen_logp_v - reference_chosen_logp_v
            v_chosen_rewards = beta * v_chosen_values.detach()

        if self.args.use_tdpo:
            logits -= self.args.tok_beta * (
                rejected_position_kl - chosen_position_kl.detach())
            chosen_values = policy_chosen_logp - reference_chosen_logp + chosen_position_kl
            rejected_values = policy_rejected_logp - reference_rejected_logp + rejected_position_kl
        else:
            chosen_values = policy_chosen_logp - reference_chosen_logp
            rejected_values = policy_rejected_logp - reference_rejected_logp

        t_losses = -F.logsigmoid(beta * logits)

        chosen_rewards = beta * chosen_values.detach()
        rejected_rewards = beta * rejected_values.detach()

        return t_losses, chosen_rewards, rejected_rewards, v_losses, q_abs_chosen, q_abs_chosen_v, q_abs_rejected, q_abs_rejected_v, v_chosen_rewards, v_rejected_rewards
    
    def get_ids(self, valid_w, target_sequence):
        #import pdb; pdb.set_trace()
        matching_indices = []
        start = None  # 第一个匹配的起始位置
        for i in range(len(valid_w) - len(target_sequence) + 1):
            if valid_w[i:i + len(target_sequence)] == target_sequence:
                if start is None:  # 第一次找到目标序列
                    start = i
                else:  # 记录上次匹配的区间，并更新start
                    matching_indices.append(list(range(start, i)))  # 记录上次的start到当前匹配位置
                    start = i  # 更新start为当前匹配的位置

        # 处理最后一个匹配
        if start is not None:
            matching_indices.append(list(range(start, len(valid_w))))  # 结束索引是最后一个有效位置

        return matching_indices

    def apply_weight_decay(self, w_mod, r_mod, win_token_weight, rej_token_weight, decay_factor, batch_idx):
        """
        Apply weight decay to the given `w_mod` and `r_mod` lists with the specified `decay_factor`.

        Args:
            w_mod (list of lists): List of valid "win" input token IDs.
            r_mod (list of lists): List of valid "rej" input token IDs.
            win_token_weight (torch.Tensor): Tensor holding the weights for "win" tokens.
            rej_token_weight (torch.Tensor): Tensor holding the weights for "rej" tokens.
            decay_factor (float): Decay factor for weight reduction.

        Returns:
            torch.Tensor, torch.Tensor: Updated win_token_weight and rej_token_weight.
        """
        # import pdb; pdb.set_trace()
        for idx in range(len(w_mod)):
            weight = self.args.dpo_token_weight * (decay_factor ** idx)  # Apply decay to the weight
            
            # Apply weight decay to w_mod (win tokens)
            win_token_weight[batch_idx][w_mod[idx]] = weight
            
            # Apply weight decay to r_mod (rejected tokens)
            rej_token_weight[batch_idx][r_mod[idx]] = weight
        
        return win_token_weight, rej_token_weight

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        data_dict = inputs

        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')
        images = data_dict.pop('images')
        diffusion_image = data_dict.pop('diffusion_image', '')
        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        assert win_size == rej_size

        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
        
        no_img_concatenated_input_ids = data_dict.pop('no_img_concatenated_input_ids')
        no_img_concatenated_labels = data_dict.pop('no_img_concatenated_labels')
        no_img_concatenated_attention_mask = data_dict.pop('no_img_concatenated_attention_mask')

        concatenated_input_ids_4 = data_dict.pop('concatenated_input_ids_4')
        concatenated_labels_4 = data_dict.pop('concatenated_labels_4')
        concatenated_attention_mask_4 = data_dict.pop('concatenated_attention_mask_4')
        image_sizes = data_dict.pop('image_sizes')
        modalities = data_dict.pop('modalities')
        diffusion_image_sizes = data_dict.pop('diffusion_image_sizes')
        diffusion_modalities = data_dict.pop('diffusion_modalities')
        causal_flag = data_dict.pop('causal')
        diffusion_idx= data_dict.pop('diffusion_idx') 
        #import pdb;pdb.set_trace()
        ref_logps = data_dict.pop('offline_ref_logits', None)
        if ref_logps is not None:
            ref_logps = torch.as_tensor(ref_logps).cuda()
        idx = data_dict.pop('idx', None)

        output_logits, new_labels = model(
            input_ids=concatenated_input_ids_4,
            labels=concatenated_labels_4,
            attention_mask=concatenated_attention_mask_4,
            images=images+images+diffusion_image+diffusion_image,
            dpo_forward=True,
            image_sizes=image_sizes+image_sizes+diffusion_image_sizes+diffusion_image_sizes,
            modalities=modalities+modalities+diffusion_modalities+diffusion_modalities,
            **data_dict
        )

        output_logits_noimage, new_labels_noimage = model(
            input_ids=no_img_concatenated_input_ids,
            labels=no_img_concatenated_labels,
            attention_mask=no_img_concatenated_attention_mask,
            dpo_forward=True,
            **data_dict
        )

        if ref_logps is None:
            with torch.no_grad():
                # concatenated_input_ids包括[yw,yl]这个地方输出有助于得到[πref(yw|mw, x)，πref(yl|mw, x)]，因为都是yw所以input_ids不用变
                ref_output = self.ref_model(
                    input_ids=concatenated_input_ids,
                    labels=concatenated_labels,
                    attention_mask=concatenated_attention_mask,
                    images=images+images,
                    image_sizes=image_sizes + image_sizes,
                    modalities=modalities + modalities,
                    **data_dict
                )

            with torch.no_grad():
                # concatenated_input_ids包括[yw,yl],这个地方输出有助于得到[πref(yw|ml, x)，πref(yl|ml, x)]，因为用不到yl，所以只会用到uncond_ref_win_logp->πref(yw|ml, x),而πref(yw|mw, x)在上面得到了，等于πref(yl|ml, x)是没用的
                unconditional_ref_output = self.ref_model(
                    input_ids=concatenated_input_ids,
                    labels=concatenated_labels,
                    attention_mask=concatenated_attention_mask,
                    images=diffusion_image+diffusion_image,
                    image_sizes=diffusion_image_sizes + diffusion_image_sizes,
                    modalities=diffusion_modalities + diffusion_modalities,
                    **data_dict
                )
            
            with torch.no_grad():
                ref_output_no_image = self.ref_model(
                    input_ids=no_img_concatenated_input_ids,
                    labels=no_img_concatenated_labels,
                    attention_mask=no_img_concatenated_attention_mask,
                    **data_dict
                )

            all_position_kl, policy_logps, policy_logps_noimage, ref_logps, uncond_ref_logps, no_image_ref_logps, \
            per_policy_token_logps, per_reference_token_logps, per_uncond_ref_token_logps, per_no_image_policy_token_logps, per_no_image_ref_token_logps = chip_get_batch_logps_symmetrical(
                output_logits, output_logits_noimage, ref_output.logits, unconditional_ref_output.logits, ref_output_no_image.logits,
                new_labels,new_labels_noimage, average_log_prob=False)

            chosen_position_kl, rejected_position_kl = all_position_kl.split([win_size, rej_size])
            uncond_ref_win_logp, uncond_ref_rej_logp = uncond_ref_logps.split([win_size, rej_size])
            no_image_ref_win_logp, no_image_ref_rej_logp = no_image_ref_logps.split([win_size, rej_size])
            no_image_policy_win_logp, no_image_policy_rej_logp = policy_logps_noimage.split([win_size, rej_size])

        # three-tuple logits,这个是正常得到policy_logps的公式的，不包括公式8~9的。
        per_policy_token_logps, policy_logps, average_policy_logps, policy_vocab_ps, loss_mask = get_batch_logps(
            output_logits, new_labels,
            return_all=True)

        # 这部分获取的是policy模型对于三元组输入的logits输出。per_policy_token_logps可以不用
        reference_chosen_logp, reference_rejected_logp = ref_logps.split([win_size, rej_size])
        
        # 文本dpo部分：就是获得ref模型对于chosen和rejected的logp

        # per-token split
        win_labels, rej_labels = concatenated_labels.split([win_size, rej_size])
        no_img_win_labels, no_img_rej_labels = no_img_concatenated_labels.split([win_size, rej_size])
        win_input_ids, rej_input_ids = concatenated_input_ids.split([win_size, rej_size])
        no_img_win_input_ids, no_img_rej_input_ids = no_img_concatenated_input_ids.split([win_size, rej_size])
        policy_chosen_logp, policy_rejected_logp, policy_win_diffusionImage_logp, policy_rej_diffusionImage_logp = policy_logps.split([win_size, rej_size, win_size, rej_size])
        policy_win_per_token_logps, policy_rej_per_token_logps,policy_win_diffusionImage_per_token_logps, policy_rej_diffusionImage_per_token_logps = per_policy_token_logps.split([win_size, rej_size, win_size, rej_size])
        ref_win_per_token_logps, ref_rej_per_token_logps = per_reference_token_logps.split([win_size, rej_size])
        uncond_ref_win_pre_token_logps, uncond_ref_rej_pre_token_logps = per_uncond_ref_token_logps.split([win_size, rej_size])
        no_image_policy_win_per_token_logps, no_image_policy_rej_per_token_logps = per_no_image_policy_token_logps.split([win_size, rej_size])
        no_image_ref_win_per_token_logps, no_image_ref_rej_per_token_logps = per_no_image_ref_token_logps.split([win_size, rej_size])
    
        #import pdb; pdb.set_trace()
        
        ########################################

        if self.args.dpo_token_weighted:
            win_token_weight, rej_token_weight = self.get_seg_weight(win_labels, rej_labels, win_input_ids, rej_input_ids)
            no_img_win_token_weight, no_img_rej_token_weight = self.get_seg_weight(no_img_win_labels, no_img_rej_labels, no_img_win_input_ids, no_img_rej_input_ids)

            uncond_ref_win_logp = self.compute_weighted_logp(uncond_ref_win_pre_token_logps, win_labels, win_token_weight)
            uncond_ref_rej_logp = self.compute_weighted_logp(uncond_ref_rej_pre_token_logps, rej_labels, rej_token_weight)

            reference_chosen_logp = self.compute_weighted_logp(ref_win_per_token_logps, win_labels, win_token_weight)
            reference_rejected_logp = self.compute_weighted_logp(ref_rej_per_token_logps, rej_labels, rej_token_weight)

            no_image_ref_win_logp = self.compute_weighted_logp(no_image_ref_win_per_token_logps, no_img_win_labels, no_img_win_token_weight)
            no_image_ref_rej_logp = self.compute_weighted_logp(no_image_ref_rej_per_token_logps, no_img_rej_labels, no_img_rej_token_weight)

            policy_chosen_logp = self.compute_weighted_logp(policy_win_per_token_logps, win_labels, win_token_weight)
            policy_rejected_logp = self.compute_weighted_logp(policy_rej_per_token_logps, rej_labels, rej_token_weight)

            policy_win_diffusionImage_logp = self.compute_weighted_logp(policy_win_diffusionImage_per_token_logps, win_labels, win_token_weight)
            policy_rej_diffusionImage_logp = self.compute_weighted_logp(policy_rej_diffusionImage_per_token_logps, rej_labels, rej_token_weight)

            no_image_policy_win_logp = self.compute_weighted_logp(no_image_policy_win_per_token_logps, no_img_win_labels, no_img_win_token_weight)
            no_image_policy_rej_logp = self.compute_weighted_logp(no_image_policy_rej_per_token_logps, no_img_rej_labels, no_img_rej_token_weight)
            
            reference_chosen_logp_v = reference_chosen_logp
            policy_chosen_logp_v = policy_chosen_logp
        else:
            reference_chosen_logp_v = reference_chosen_logp
            policy_chosen_logp_v = policy_chosen_logp

        ########################################
        policy_win_ps, policy_rej_ps, policy_win_diffusionImage_ps, policy_rej_diffusionImage_ps = policy_vocab_ps.split([win_size, rej_size, win_size, rej_size])

        dpo_losses, chosen_rewards, rejected_rewards, v_dpo_losses, q_abs_chosen, q_abs_chosen_v, q_abs_rejected, q_abs_rejected_v, v_chosen_rewards, v_rejected_rewards= self.chip_loss_symmetrical(
            policy_chosen_logp, policy_rejected_logp, 
            policy_win_diffusionImage_logp, policy_rej_diffusionImage_logp,
            no_image_policy_win_logp, no_image_policy_rej_logp,
            reference_chosen_logp, reference_rejected_logp,
            uncond_ref_win_logp, uncond_ref_rej_logp,
            no_image_ref_win_logp, no_image_ref_rej_logp,
            chosen_position_kl, rejected_position_kl,
            policy_chosen_logp_v,
            reference_chosen_logp_v,
            causal_flag=causal_flag,
            diffusion_idx=diffusion_idx
        )

        chosen_labels = new_labels[:win_size]
        chosen_logits = output_logits[:win_size]

        dpo_loss = dpo_losses.mean()
        v_dpo_loss = v_dpo_losses.mean()
        chosen_abs_loss = q_abs_chosen.mean()
        v_chosen_abs_loss= q_abs_chosen_v.mean()
        rejected_abs_loss = q_abs_rejected.mean()
        v_rejected_abs_loss = q_abs_rejected_v.mean()
        unscaled_sft_losses = self.get_sft_loss(chosen_logits, chosen_labels)
        unscaled_sft_loss = unscaled_sft_losses.mean()
        sft_loss = unscaled_sft_loss * 0.1

        if self.args.use_symmetrical_loss:
            if self.args.use_cross_modal_loss:
                loss = dpo_loss + v_dpo_loss + sft_loss
            else:
                loss = v_dpo_loss + sft_loss
        else:
            loss = dpo_loss + sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        v_reward_accuracies = (v_chosen_rewards > v_rejected_rewards).float()
        def all_gather_tensor(tensor):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tensor = tensor.detach()
                gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered_tensor, tensor)
                tensor = torch.cat(gathered_tensor, dim=0)
            return tensor

        # gather chosen_rewards across devices
        chosen_rewards = all_gather_tensor(chosen_rewards)
        v_chosen_rewards = all_gather_tensor(v_chosen_rewards)
        rejected_rewards = all_gather_tensor(rejected_rewards)
        v_rejected_rewards = all_gather_tensor(v_rejected_rewards)
        reward_accuracies = all_gather_tensor(reward_accuracies)
        v_reward_accuracies = all_gather_tensor(v_reward_accuracies)
        policy_chosen_logps = all_gather_tensor(policy_chosen_logp)
        policy_chosen_logps_v = all_gather_tensor(policy_chosen_logp_v)
        policy_rejected_logps = all_gather_tensor(policy_rejected_logp)
        reference_chosen_logps_v = all_gather_tensor(reference_chosen_logp_v)
        policy_win_diffusionImage_logps = all_gather_tensor(policy_win_diffusionImage_logp)
        reference_win_diffusionImage_logps = all_gather_tensor(uncond_ref_win_logp)
        reference_chosen_logps = all_gather_tensor(reference_chosen_logp)
        reference_rejected_logps = all_gather_tensor(reference_rejected_logp)

        prefix = "" if model.training else 'eval_'
        metrics = {}
        metrics[f"{prefix}losses/v_dpo"] = v_dpo_loss.cpu()
        metrics[f"{prefix}losses/v_rejected_abs"] = v_rejected_abs_loss.cpu()
        metrics[f"{prefix}losses/v_chosen_abs"] = v_chosen_abs_loss.cpu()

        metrics[f"{prefix}losses/dpo"] = dpo_loss.cpu()
        metrics[f"{prefix}losses/chosen_abs"] = chosen_abs_loss.cpu()
        metrics[f"{prefix}losses/rejected_abs"] = rejected_abs_loss.cpu()

        metrics[f"{prefix}losses/sft"] = unscaled_sft_loss.cpu()
        metrics[f"{prefix}losses/total"] = loss.cpu()
        # rewards
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/v_rejected"] = v_rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/v_chosen"] = v_chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/v_accuracies"] = v_reward_accuracies.mean().cpu()
        # margins
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}rewards/v_margins"] = (v_chosen_rewards - v_rejected_rewards).mean().cpu()
        # policy logps
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/v_chosen"] = policy_chosen_logps_v.detach().mean().cpu()
        metrics[f"{prefix}logps/v_rejected"] = policy_win_diffusionImage_logps.detach().mean().cpu()
        # reference logps
        metrics[f"{prefix}ref_logps/rejected"] = reference_rejected_logps.mean().cpu()
        metrics[f"{prefix}ref_logps/chosen"] = reference_chosen_logps.mean().cpu()
        metrics[f"{prefix}ref_logps/v_chosen"] = reference_chosen_logps_v.mean().cpu()
        metrics[f"{prefix}ref_logps/v_rejected"] = reference_win_diffusionImage_logps.detach().mean().cpu()
        self.store_metrics(metrics, train_eval="train")
        return loss

    def get_seg_weight(self, win_labels, rej_labels,win_input_ids, rej_input_ids):
        win_token_weight = torch.ones_like(win_labels[:, 1:], dtype=torch.bfloat16)
        rej_token_weight = torch.ones_like(rej_labels[:, 1:], dtype=torch.bfloat16)
        for idx, (w, r) in enumerate(zip(win_input_ids, rej_input_ids)):
            valid_w = w[1:]
            valid_r = r[1:]
            min_match_size = 3
            r_mod, w_mod = get_diff_ids(valid_r.tolist(), valid_w.tolist(), min_match_size=min_match_size)
            win_token_weight[idx][w_mod] = self.args.dpo_token_weight
            rej_token_weight[idx][r_mod] = self.args.dpo_token_weight

        return win_token_weight, rej_token_weight
    
    @staticmethod    
    def compute_weighted_logp(per_token_logp, labels, token_weight, use_average=False):
        loss_mask = (labels[:, 1:].clone() != -100)
        weighted_mask = token_weight * loss_mask
        if len(per_token_logp.shape)!=1:
            per_token_logp = per_token_logp[:, -weighted_mask.shape[1]:]
        logp = (per_token_logp * weighted_mask).sum(-1)

        average_logp = logp / weighted_mask.sum(-1)
        if use_average:
            return average_logp

        return logp


    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explictly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[[prompt, pol[len(prompt) :], ref[len(prompt) :]] for prompt, pol, ref in zip(random_batch["prompt"], policy_output_decoded, ref_output_decoded)],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "sft" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)

