#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Jiawei Liu on 2018/06/18

import os

from oauth2client.client import GoogleCredentials
from six.moves import input  # pylint: disable=redefined-builtin

from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.serving import serving_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import tensorflow as tf


def make_request_fn(server_name, server_address, timeout_secs=10):
    """Returns a request function."""
    request_fn = serving_utils.make_grpc_request_fn(
        servable_name=server_name,
        server=server_address,
        timeout_secs=timeout_secs)
    return request_fn


def query_t2t(input_txt, data_dir, problem_name, server_name, server_address, t2t_usr_dir):
    usr_dir.import_usr_dir(t2t_usr_dir)
    problem = registry.problem(problem_name)
    hparams = tf.contrib.training.HParams(data_dir=os.path.expanduser(data_dir))
    problem.get_hparams(hparams)
    request_fn = make_request_fn(server_name, server_address)
    inputs = input_txt
    outputs = serving_utils.predict([inputs], problem, request_fn)
    output, score = outputs
    print(output)
    return output, score


def main():
    input_txt = "▁Human s ▁have ▁many ▁basic ▁needs ▁and ▁one ▁of ▁them ▁is ▁to ▁have ▁an ▁environment ▁that ▁can ▁sustain ▁their ▁lives ▁ ."
    problem_name = "gec_pretrain_transformer"
    data_dir = "/data/liujiawei/home_backup/gitlab/pretrain_model/data_dir"
    t2t_usr_dir = "/data/liujiawei/home_backup/gitlab/pretrain_model"
    server_address = "127.0.0.1:8500"
    server_name = "pretrain_gec_model"
    query_t2t(input_txt,
              data_dir,
              problem_name,
              server_name,
              server_address,
              t2t_usr_dir)


if __name__ == "__main__":
    main()
