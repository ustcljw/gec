#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Jiawei Liu on 2018/06/18

from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_base, transformer_big

@registry.register_hparams
def transformer_base_gec():
    hparams = transformer_base()
    hparams.max_length = 150
    hparams.batch_size = 2048
    hparams.learning_rate = 0.0003
    hparams.learning_rate_warmup_steps = 16000
    hparams.layer_prepostprocess_dropout = 0.3
    hparams.attention_dropout = 0.1
    hparams.relu_dropout = 0.1
    hparams.label_smoothing = 0.1
    hparams.learning_rate_schedule = ("constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size") #tensor2tensor源码设置，可以通过此修改learning rate的warm up和decay
    hparams.shared_embedding_and_softmax_weights = True

    return hparams


@registry.register_hparams
def transformer_big_gec():
    hparams = transformer_big()
    hparams.max_length = 150
    hparams.batch_size = 2048
    hparams.learning_rate = 0.0002
    hparams.learning_rate_warmup_steps = 8000
    hparams.layer_prepostprocess_dropout = 0.3
    hparams.attention_dropout = 0.1
    hparams.relu_dropout = 0.1
    hparams.label_smoothing = 0.1
    hparams.learning_rate_schedule = ("constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
    hparams.shared_embedding_and_softmax_weights = True

    return hparams