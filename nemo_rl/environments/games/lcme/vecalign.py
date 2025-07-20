#!/usr/bin/env python3

"""
Copyright 2019 Brian Thompson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import logging
import pickle
from math import ceil
from random import seed as seed

import numpy as np

logger = logging.getLogger('vecalign')
logger.setLevel(logging.WARNING)
logFormatter = logging.Formatter("%(asctime)s  %(levelname)-5.5s  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

from nemo_rl.environments.games.lcme.dp_utils import make_alignment_types, make_one_to_many_alignment_types, print_alignments, read_alignments, \
    read_in_embeddings, make_doc_embedding, vecalign

from nemo_rl.environments.games.lcme.score import score_multiple, log_final_scores


class VecalignArgs:
    def __init__(self):
        # Arguments from the original parser
        self.gold_alignment = None
        self.one_to_many = None
        self.verbose = False
        self.max_size_full_dp = 300
        self.costs_sample_size = 20000
        self.num_samps_for_norm = 100
        self.search_buffer_size = 5
        self.debug_save_stack = None
        self.print_aligned_text = False
        
        # Additional arguments referenced in the code
        self.alignment_max_size = 2
        self.del_percentile_frac = 0.1
        self.src = []
        self.tgt = []
        self.src_embed = []
        self.tgt_embed = []
        self.output_file = None


def _main(override_args=None):
    # make runs consistent
    seed(42)
    np.random.seed(42)

    args_obj = VecalignArgs()
    if override_args is not None:
        for key, value in override_args.items():
            setattr(args_obj, key, value)
    args = args_obj

    if len(args.src) != len(args.tgt):
        raise Exception('number of source files must match number of target files')

    if args.gold_alignment is not None:
        if len(args.gold_alignment) != len(args.src):
            raise Exception('number of gold alignment files, if provided, must match number of source and target files')

    if args.verbose:
        import logging
        logger.setLevel(logging.INFO)

    if args.alignment_max_size < 2:
        logger.warning('Alignment_max_size < 2. Increasing to 2 so that 1-1 alignments will be considered')
        args.alignment_max_size = 2

    src_sent2line, src_line_embeddings = read_in_embeddings(args.src_embed[0], args.src_embed[1])
    tgt_sent2line, tgt_line_embeddings = read_in_embeddings(args.tgt_embed[0], args.tgt_embed[1])

    src_max_alignment_size = 1 if args.one_to_many is not None else args.alignment_max_size
    tgt_max_alignment_size = args.one_to_many if args.one_to_many is not None else args.alignment_max_size

    width_over2 = ceil(max(src_max_alignment_size, tgt_max_alignment_size) / 2.0) + args.search_buffer_size

    test_alignments = []
    stack_list = []

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for src_file, tgt_file in zip(args.src, args.tgt):
            logger.info('Aligning src="%s" to tgt="%s"', src_file, tgt_file)

            src_lines = open(src_file, 'rt', encoding="utf-8").readlines()
            vecs0 = make_doc_embedding(src_sent2line, src_line_embeddings, src_lines, src_max_alignment_size)

            tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
            vecs1 = make_doc_embedding(tgt_sent2line, tgt_line_embeddings, tgt_lines, tgt_max_alignment_size)

            if args.one_to_many is not None:
                final_alignment_types = make_one_to_many_alignment_types(args.one_to_many)
            else:
                final_alignment_types = make_alignment_types(args.alignment_max_size)
            logger.debug('Considering alignment types %s', final_alignment_types)

            stack = vecalign(vecs0=vecs0,
                            vecs1=vecs1,
                            final_alignment_types=final_alignment_types,
                            del_percentile_frac=args.del_percentile_frac,
                            width_over2=width_over2,
                            max_size_full_dp=args.max_size_full_dp,
                            costs_sample_size=args.costs_sample_size,
                            num_samps_for_norm=args.num_samps_for_norm)

            # write final alignments to stdout
            print_alignments(stack[0]['final_alignments'], scores=stack[0]['alignment_scores'],
                            src_lines=src_lines if args.print_aligned_text else None,
                            tgt_lines=tgt_lines if args.print_aligned_text else None,
                            ofile=f)

            test_alignments.append(stack[0]['final_alignments'])
            stack_list.append(stack)

    if args.gold_alignment is not None:
        gold_list = [read_alignments(x) for x in args.gold_alignment]
        res = score_multiple(gold_list=gold_list, test_list=test_alignments)
        log_final_scores(res)

    if args.debug_save_stack:
        pickle.dump(stack_list, open(args.debug_save_stack, 'wb'))


if __name__ == '__main__':
    _main()
