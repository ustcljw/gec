#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Jiawei Liu on 2019/06/18

import os

import tensorflow as tf
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems, text_encoder
from tensor2tensor.utils import registry

GEC_DATASETS = {
    "TRAIN": "all.pieces",
    "DEV": "wi_locness.NABC.dev.pieces",
    "VOCAB": "vocab.txt"
}


@registry.register_problem
class GecPretrainTransformer(text_problems.Text2TextProblem):

    @property
    def approx_vocab_size(self):
        return 2 ** 15

    @property
    def is_generate_per_split(self):
        return True

    # @property
    # def dataset_splits(self):
    #     return [{
    #         "split": problem.DatasetSplit.TRAIN,
    #         "shards": 100,
    #     }, {
    #         "split": problem.DatasetSplit.EVAL,
    #         "shards": 1,
    #     }]

    def get_vocab(self, data_dir):
        vocab_path = os.path.join(data_dir, GEC_DATASETS["VOCAB"])
        if not tf.gfile.Exists(vocab_path):
            raise ValueError("vocab file is not found")
        return text_encoder.TokenTextEncoder(vocab_filename=vocab_path, replace_oov="<unk>")

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        train_path = os.path.join(tmp_dir, "oversampled", GEC_DATASETS["TRAIN"]) if train else os.path.join(tmp_dir, "wi_locness", GEC_DATASETS["DEV"])
        vocab_path = os.path.join(tmp_dir, GEC_DATASETS["VOCAB"])
        vocab_path_src_tgt = os.path.join(data_dir, GEC_DATASETS["VOCAB"])

        if not tf.gfile.Exists(vocab_path_src_tgt):
            tf.gfile.Copy(vocab_path, vocab_path_src_tgt)
            with tf.gfile.GFile(vocab_path_src_tgt, mode="r") as fr:
                vocab_data = "<pad>\n<EOS>\n" + fr.read()
                with tf.gfile.GFile(vocab_path_src_tgt, mode="w") as fw:
                    fw.write(vocab_data)
        return text_problems.text2text_txt_iterator(train_path + ".src",
                                                    train_path + ".tgt")

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_vocab(data_dir)
        return text_problems.text2text_generate_encoded(generator, encoder, encoder, has_inputs=self.has_inputs)

    def feature_encoders(self, data_dir):
        src_token = self.get_vocab(data_dir)
        tgt_token = self.get_vocab(data_dir)
        return {
            "inputs": src_token,
            "targets": tgt_token
        }


