# coding:utf-8

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import copy
import json
import torch
import codecs
import functools

from typing import List
from torch.utils.data import DataLoader
from utils import str_to_num

class DataBatchGenerator:
    """Generate Data Batch."""
    def __init__(self, args: dict, mode: str, op_list: List, const_list: List):
        self.args = copy.deepcopy(args)
        self.mode = mode
        self.op_list = op_list
        self.const_list = const_list

        if args.plm.startswith("roberta"):
            from transformers import RobertaTokenizer
            from transformers import RobertaConfig
            self.tokenizer = RobertaTokenizer.from_pretrained(self.args.plm)
            self.plm_config = RobertaConfig.from_pretrained(self.args.plm)
            self.max_position_embeddings = 512
            self.sep_token = self.tokenizer.sep_token
            self.offset = 1
        elif args.plm.startswith("t5"):
            from transformers import T5Tokenizer
            from transformers import T5Config
            self.tokenizer = T5Tokenizer.from_pretrained(self.args.plm)
            self.plm_config = T5Config.from_pretrained(self.args.plm)
            self.max_position_embeddings = self.plm_config.n_positions
            self.sep_token = self.tokenizer.eos_token
            self.offset = 0
        else:
            raise NotImplementedError
        self.plm_name = args.plm

        self.data_name = self.args.data_name

        if self.mode == "train":
            data_path = self.args.cached_train_data
        elif self.mode == "eval":
            data_path = self.args.cached_dev_data
        elif self.mode == "test":
            data_path = self.args.cached_test_data
        else:
            raise ValueError(f"unknown mode: {self.mode}.")

        # FIXME: this is for temporary calculating the average length
        self.operation_length = []
        self.program_steps_count = {}

        with codecs.open(data_path, "r") as file:
            instances = json.load(file)
        
        print(f"[INFO]: start to initialize data...")
        self._init_data(instances, self.mode)
        print(f"INFO: succeed in initializing data with size: {len(self.datas)} ")
        print(f"Global maximize operation length: {self.max_op_len}")
        print(f"Global maximize op_argument length: {self.max_op_argu_len}")
        print(f"Global maximize program length: {self.max_prog_len}")
        
        self.program_steps_count = dict(sorted(self.program_steps_count.items(), key=lambda c: c[0]))
        for op_len, count in self.program_steps_count.items():
            print(f"Program Steps: {op_len} has {count} data.")

    def _init_data(self, instances, mode):
        """convert the token to ids, program to ids."""
        self.datas = []
        # NOTE: I thought to batch the op or argu according to their maximize length
        #       in a batch, however, that will lead to the loss cannot be calculated
        #       when doing evaluation. This is because the golden_op_max_len is
        #       different with the inference.
        self.max_op_len = 0
        self.max_op_argu_len = 0
        self.max_prog_len = 0
        for instance in instances:
            # question_passage_tokens -> question_passage_ids
            question_passage_tokens = instance["question_passage_tokens"]
            if self.plm_name.startswith("roberta"):
                if len(question_passage_tokens) > (self.max_position_embeddings - 2):
                    question_passage_tokens = question_passage_tokens[: self.max_position_embeddings - 2]
                question_passage_tokens = [self.tokenizer.cls_token] + question_passage_tokens + [self.tokenizer.sep_token]
            elif self.plm_name.startswith("t5"):
                if len(question_passage_tokens) > (self.max_position_embeddings - 1):
                    question_passage_tokens = question_passage_tokens[: self.max_position_embeddings - 1]           
                question_passage_tokens = question_passage_tokens + [self.sep_token]
            else:
                raise ValueError(f"Unknown PLM: {self.plm_name}")

            question_passage_ids = self.tokenizer.convert_tokens_to_ids(question_passage_tokens)
            question_index = question_passage_tokens.index(self.sep_token)

            # offset the number_indices because of [cls] in front of the question_passage_tokens
            argument_indices = instance["arguments_indices"]
            argument_indices_offset = [idx + self.offset for idx in argument_indices]
            instance["argument_indices_offset"] = argument_indices_offset
            
            # if self.data_name == "finqa":
            #     number_indices = instance["number_indices"]
            #     number_indices_offset = [idx + self.offset for idx in number_indices]
            #     instance["number_indices"] = number_indices_offset

            #     number_number_graph_mask = [[0 for _ in range(len(question_passage_ids))] for _ in range(len(number_indices_offset))]
            #     for sub_mask in number_number_graph_mask:
            #         for ind in number_indices_offset:
            #             sub_mask[ind] = 1
            # else:
            number_indices_offset = None
            number_number_graph_mask = None

            # create mask for the numbers in the question_passage_tokens
            number_in_qp_mask = [0 for _ in range(len(question_passage_ids))]
            for ids in instance["argument_indices_offset"]:
                number_in_qp_mask[ids] = 1
            
            # if self.data_name == "finqa":
            #     number_unit_graph = instance["number_unit_graph"]
            #     number_unit_graph_offset = {int(key) + self.offset: value for key, value in number_unit_graph.items()}
            #     instance["number_unit_graph"] = number_unit_graph_offset

            #     # create unit graph mask
            #     number_unit_graph_mask = [[0 for _ in range(6)] for _ in range(len(number_indices))]
            #     for i, key_value in enumerate(number_unit_graph_offset.items()):
            #         unit_idx = key_value[1]
            #         if unit_idx != 0:
            #             number_unit_graph_mask[i][unit_idx] = 1
            # else:
            number_unit_graph_mask = None

            # if self.mode in ["train", "eval"]:
            if self.mode in ["train"]:
                # golden_op -> golden_op_ids
                golden_op_ids = [self.op_list.index(op) for op in instance["golden_op"]]
    
                # update the maximize gold_op_len in the total datas
                gold_op_len = len(golden_op_ids)
                if gold_op_len > self.max_op_len:
                    self.max_op_len = gold_op_len

                try:
                    # golden_argument -> golden_argument_ids
                    golden_argument_ids = self.convert_argu_to_ids(instance["golden_argument"],
                                                                instance["arguments"],
                                                                instance["argument_indices_offset"],
                                                                instance["id"])
                except ValueError:
                    print(instance["id"])
                    print(instance["golden_argument"])
                    print(instance["arguments"])
                    print(instance["argument_indices_offset"])
                    exit()
                
                
                gold_op_argu_len = max([len(op_argu) for op_argu in instance["golden_argument"]])
                if gold_op_argu_len > self.max_op_argu_len:
                    self.max_op_argu_len = gold_op_argu_len
                
                self.operation_length.append(gold_op_len)
                if gold_op_len in self.program_steps_count:
                    self.program_steps_count[gold_op_len] +=1
                else:
                    self.program_steps_count[gold_op_len] = 1

                # NOTE: [v5.0: for baseline FinQA on MathQA]
                if bool(self.args.is_program_as_sequence):
                    self.op_argu_list = self.op_list + self.const_list
                    sequence_program = instance["sequence_program"]
                    golden_sequence_program_ids = self.convert_op_argu_to_ids(
                        sequence_program,
                        instance["arguments"],
                        instance["argument_indices_offset"]
                    )
                    if len(sequence_program) > self.max_prog_len:
                        self.max_prog_len = len(sequence_program)
                else:
                    golden_sequence_program_ids = None
            else:
                golden_op_ids = None
                golden_argument_ids = None
                golden_sequence_program_ids = None
            

            units_token = [
                self.tokenizer.tokenize("<pad>"),
                self.tokenizer.tokenize("ousand"),
                self.tokenizer.tokenize("million"),
                self.tokenizer.tokenize("billion"),
                self.tokenizer.tokenize("percent"),
                self.tokenizer.tokenize("year")
            ]
            units_idx = [self.tokenizer.convert_tokens_to_ids(token)[0] for token in units_token]

            self.datas.append((question_passage_ids, 
                               question_index, 
                               number_unit_graph_mask, 
                               number_number_graph_mask, 
                               number_indices_offset,
                               number_in_qp_mask, 
                               golden_op_ids, 
                               golden_argument_ids,
                               units_idx, 
                               golden_sequence_program_ids,
                               instance))
    
    def convert_argu_to_ids(self,
                            golden_argument, 
                            arguments, 
                            argument_indices_offset,
                            id=None):
        golden_argument_ids = []
        for sub_argu in golden_argument:
            sub_argu_ids = []
            for argu in sub_argu:
                if argu in self.const_list:
                    sub_argu_ids.append(self.const_list.index(argu))
                else:
                    if self.data_name == "finqa":
                        try:
                            idx_for_indices = arguments.index(str_to_num(argu))
                        except ValueError:
                            # because argu could be the row[0], when applying str_to_num,
                            # it will return None, we need to check again.
                            idx_for_indices = arguments.index(argu)
                    elif self.data_name == "mathqa":
                        idx_for_indices = int(argu.replace("n", ""))
                        assert idx_for_indices < len(arguments)
                    elif self.data_name in ["drop_annotated", "drop_fewshot", "drop_fakedata"]:
                        idx_for_indices = arguments.index(float(argu))
                    argu_ids = argument_indices_offset[idx_for_indices] + len(self.const_list)
                    sub_argu_ids.append(argu_ids)
            golden_argument_ids.append(sub_argu_ids)
        
        return golden_argument_ids
    
    def convert_op_argu_to_ids(self,
                               program_sequence,
                               arguments,
                               argument_indices_offset):
        golden_program_ids = []
        for tok in program_sequence:
            if tok in self.op_argu_list:
                idx = self.op_argu_list.index(tok)
            else:
                idx_in_argus = int(tok.replace("n", ""))
                assert idx_in_argus < len(arguments)
                idx = argument_indices_offset[idx_in_argus] + len(self.op_argu_list)
            golden_program_ids.append(idx)

        return golden_program_ids

    @staticmethod
    def create_batch_data(batch,
                          pad_ids, eof_ids, argu_pad_ids, 
                          max_input_len, max_op_len, max_op_argu_len, max_prog_len, offset,
                          use_cuda, mode):
        question_passage_ids_set, question_index, number_unit_graph_mask, number_number_graph_mask, number_indices_offset, number_in_qp_mask_set, golden_op_ids_set, golden_argument_ids_set, units_idx, golden_sequence_program_ids, instance_set = zip(*batch)
        bsz = len(question_passage_ids_set)

        # number graph
        if any(number_unit_graph_mask):
            max_number_of_numbers = max([len(argus) for argus in number_unit_graph_mask])
            input_number_unit_graph = torch.FloatTensor(bsz, max_number_of_numbers, 6).fill_(0)
            input_number_number_graph = torch.FloatTensor(bsz, max_number_of_numbers, max_input_len).fill_(0)
            graph_number_mask = torch.LongTensor(bsz, max_number_of_numbers).fill_(0)
  
            input_unit_ids = torch.LongTensor([units_idx[0]])
            input_number_indices = torch.LongTensor(bsz, max_number_of_numbers).fill_(-1)
        else:
            max_number_of_numbers = None
            input_number_unit_graph = None
            input_number_number_graph = None
            graph_number_mask = None

            input_unit_ids = None
            input_number_indices = None

        # [bsz, max_input_len]
        tensor_input_ids = torch.LongTensor(bsz, max_input_len).fill_(pad_ids)
        input_mask = torch.LongTensor(bsz, max_input_len).fill_(0)
        input_question_mask = torch.LongTensor(bsz, max_input_len).fill_(0)
        input_number_mask = torch.LongTensor(bsz, max_input_len).fill_(0)

        # if mode in ["train", "eval"]:
        if mode in ["train"]:
            # [bsz, max_op_in_batch]
            # FIXED: now use global maximize length
            # # max_op_in_batch = max([len(op_ids) for op_ids in golden_op_ids_set])
            tensor_op_ids = torch.LongTensor(bsz, max_op_len).fill_(eof_ids)
            golden_op_mask = torch.LongTensor(bsz, max_op_len).fill_(0)

            # [bsz, max_op_in_batch, max_argu_in_batch]
            # FIXED: now use global maximize length
            # # max_argu_in_batch = max([len(sub_argu) for argu_ids in golden_argument_ids_set for sub_argu in argu_ids])
            tensor_argu_ids = torch.LongTensor(bsz, max_op_len, max_op_argu_len).fill_(argu_pad_ids)
            golden_argu_mask = torch.LongTensor(bsz, max_op_len, max_op_argu_len).fill_(0)

            if golden_sequence_program_ids[0] != None:
                tensor_program_ids = torch.LongTensor(bsz, max_prog_len).fill_(eof_ids)
                golden_program_mask = torch.LongTensor(bsz, max_prog_len).fill_(0)
            else:
                tensor_program_ids = None
                golden_program_mask = None
        else:
            tensor_op_ids = None
            golden_op_mask = None
            tensor_argu_ids = None
            golden_argu_mask = None

            tensor_program_ids = None
            golden_program_mask = None
            

        for i in range(bsz):
            cur_input_len = len(question_passage_ids_set[i])
            tensor_input_ids[i, :cur_input_len] = torch.LongTensor(question_passage_ids_set[i])
            input_mask[i, :cur_input_len] = 1
            input_question_mask[i, offset : question_index[i]] = 1
            input_number_mask[i, :cur_input_len] = torch.LongTensor(number_in_qp_mask_set[i])

            if any(number_unit_graph_mask):
                cur_number_nums = len(number_unit_graph_mask[i])

                input_number_unit_graph[i, :cur_number_nums, :] = torch.FloatTensor(number_unit_graph_mask[i])
                input_number_number_graph[i, :cur_number_nums, :cur_input_len] = torch.FloatTensor(number_number_graph_mask[i])
                graph_number_mask[i, :cur_number_nums] = 1
                input_number_indices[i, :cur_number_nums] = torch.LongTensor(number_indices_offset[i])

            # if mode in ["train", "eval"]:
            if mode in ["train"]:
                cur_op_len = len(golden_op_ids_set[i])
                tensor_op_ids[i, :cur_op_len] = torch.LongTensor(golden_op_ids_set[i])
                golden_op_mask[i, :cur_op_len] = 1

                for sub_ids, sub_argu in enumerate(golden_argument_ids_set[i]):
                    if sub_argu[0] != argu_pad_ids:
                        cur_sub_argu_len = len(sub_argu)
                        tensor_argu_ids[i, sub_ids, :cur_sub_argu_len] = torch.LongTensor(sub_argu)
                        # golden_argu_mask[i, sub_ids, :cur_sub_argu_len] = 1
                        golden_argu_mask[i, sub_ids, :cur_sub_argu_len + 1] = 1
                
                if golden_sequence_program_ids[0] != None:
                    cur_prog_len = len(golden_sequence_program_ids[i])
                    tensor_program_ids[i, :cur_prog_len] = torch.LongTensor(golden_sequence_program_ids[i])
                    # FIXME: for finqa, op with exact 2 argus,
                    # golden_argu_mask[i, sub_ids, :cur_sub_argu_len] = 1
                    # FIXME: for MathQA, DROP, op with arbitraty number of argus.
                    golden_argu_mask[i, sub_ids, :cur_sub_argu_len + 1] = 1

        segment_ids = torch.LongTensor(bsz, max_input_len).fill_(0)

        outputs = {
            "input_ids": tensor_input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "input_number_unit_graph": input_number_unit_graph,
            "input_number_number_graph": input_number_number_graph,
            "input_number_indices": input_number_indices,
            "graph_number_mask": graph_number_mask,
            "input_unit_ids": input_unit_ids,
            "input_question_mask": input_question_mask,
            "input_number_mask": input_number_mask,
            "golden_op": tensor_op_ids,
            "golden_op_mask": golden_op_mask,
            "golden_arguments": tensor_argu_ids,
            "golden_argu_mask": golden_argu_mask,
            "golden_program": tensor_program_ids,
            "golden_program_mask": golden_program_mask,
            "metadata": instance_set,
        }

        if use_cuda:
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    outputs[k] = v.cuda()
        
        return outputs

    def data_loader(self, use_cuda):
        collate_fn = functools.partial(
            self.create_batch_data,
            pad_ids=self.tokenizer.pad_token_id,
            eof_ids=self.op_list.index("EOF"),
            argu_pad_ids=self.const_list.index("none"),
            max_input_len=self.max_position_embeddings,
            max_op_len=self.max_op_len,
            max_op_argu_len=self.max_op_argu_len,
            max_prog_len=self.max_prog_len,
            offset=self.offset,
            use_cuda=use_cuda,
            mode=self.mode
        )

        loader = DataLoader(
            self.datas,
            batch_size=self.args.t_bsz if self.mode == "train" else self.args.e_bsz,
            shuffle=self.mode == "train",
            collate_fn=collate_fn,
        )

        return loader
    
    def data_loader_kfold(self, use_cuda, fold_data, shuffle_or_not):
        collate_fn = functools.partial(
            self.create_batch_data,
            pad_ids=self.tokenizer.pad_token_id,
            eof_ids=self.op_list.index("EOF"),
            argu_pad_ids=self.const_list.index("none"),
            max_input_len=self.max_position_embeddings,
            max_op_len=self.max_op_len,
            max_op_argu_len=self.max_op_argu_len,
            max_prog_len=self.max_prog_len,
            offset=self.offset,
            use_cuda=use_cuda,
            mode=self.mode
        )

        loader = DataLoader(
            fold_data,
            batch_size=self.args.t_bsz,
            shuffle=shuffle_or_not,
            collate_fn=collate_fn
        )
        
        return loader
    
    def __len__(self):
        batch_number = len(self.datas) // self.args.t_bsz
        batch_number = batch_number if len(self.datas) % self.args.t_bsz == 0 else batch_number + 1
        return batch_number

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--plm")
    parser.add_argument("--cached_train_data")
    parser.add_argument("--cached_dev_data")
    parser.add_argument("--cached_test_data")
    parser.add_argument("--data_name")
    parser.add_argument("--is_program_as_sequence")
    args = parser.parse_args()

    args.data_name = "mathqa"
    args.plm = "roberta-large"
    args.cached_train_data = f"../datasets/cached_data/{args.data_name}/cached_train_data.json"
    args.cached_dev_data = f"../datasets/cached_data/{args.data_name}/cached_dev_data.json"
    args.cached_test_data = f"../datasets/cached_data/{args.data_name}/cached_test_data.json"
    args.is_program_as_sequence = 0

    args.t_bsz = 2
    args.e_bsz = 16

    parser_2 = ArgumentParser()
    parser_2.add_argument("--plm")
    parser_2.add_argument("--cached_train_data")
    parser_2.add_argument("--cached_dev_data")
    parser_2.add_argument("--cached_test_data")
    parser_2.add_argument("--data_name")
    parser_2.add_argument("--is_program_as_sequence")
    args_2 = parser_2.parse_args()

    args_2.data_name = "finqa"
    args_2.plm = "roberta-large"
    args_2.cached_train_data = f"../datasets/cached_data/{args_2.data_name}/cached_train_data.json"
    args_2.cached_dev_data = f"../datasets/cached_data/{args_2.data_name}/cached_dev_data.json"
    args_2.cached_test_data = f"../datasets/cached_data/{args_2.data_name}/cached_test_data.json"
    args_2.is_program_as_sequence = 0

    args_2.t_bsz = 2
    args_2.e_bsz = 16

    CONST_LIST_FINQA = ["const_1", "const_2", "const_3", "const_4", "const_5", "const_6", "const_7", "const_8", "const_9", "const_10",
                "const_100", "const_1000", "const_10000", "const_100000", "const_1000000", "const_10000000",
                "const_1000000000", "const_m1", "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10", 
                "none"]

    OPERATION_LIST_FINQA = ["add", "subtract", "multiply", "divide", "exp", "greater", 
                    "table_sum", "table_average", "table_max", "table_min", "biggest", "smallest",
                    "EOF"]

    CONST_LIST = ["const_pi", "const_2", "const_1", "const_3", "const_4", "const_6", "const_10", "const_12", "const_100", "const_1000",
                        "const_60", "const_3600", "const_1.6", "const_0.6", "const_0.2778", "const_0.3937", "const_2.54",
                        "const_0.4535", "const_2.2046", "const_3.6", "const_deg_to_rad", "const_180", "const_0.25", "const_0.33",
                        "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10", "#11", "#12", "#13", "#14", "#15", "#16",
                        "none"]

    OPERATION_LIST = ["add", "subtract", "multiply", "divide", "gcd", "lcm", "power", "max", "min",
                            "reminder", "round", "radians_to_degress", "degree_to_radians",
                            "sum_consecutive_number", "circle_arc", "semi_circle_perimiter", "circle_sector_area",
                            "rectangle_perimeter", "rectangle_area", "trapezium_area",
                            "rhombus_area", "quadrilateral_area", "volume_cone", "volume_rectangular_prism",
                            "volume_cylinder", "surface_cone", "surface_cylinder", "surface_rectangular_prism",
                            "side_by_diagonal", "diagonal", "triangle_perimeter",
                            "triangle_area", "triangle_area_three_edges", "union_prob", "combination", "permutation", "count_interval",
                            "percent", "p_after_gain", "p_after_loss", "price_after_gain", "price_after_loss", "from_percent", "gain_percent",
                            "loss_percent", "negate_percent", "original_price_before_gain", "original_price_before_loss", "to_percent", "speed",
                            "combined_work", "find_work", "speed_ratio_steel_to_stream", "speed_in_still_water", "stream_speed", 
                            "floor", "cosine",
                            "cube_edge_by_volume", "volume_cube", "sine", "factorial", "square_area", "negate", "sqrt", "circle_area",  "surface_sphere",
                            "log", "surface_cube", "rhombus_perimeter", "volume_sphere", "tangent", "square_perimeter", "circumface", "square_edge_by_area",
                            "inverse", "square_edge_by_perimeter", "negate_prob",
                            "GO", ")", "EOF"]


    train_data_batch_generator = DataBatchGenerator(args, "train", OPERATION_LIST, CONST_LIST)
    # dev_data_batch_generator = DataBatchGenerator(args, "eval", OPERATION_LIST, CONST_LIST)
    # FIXME: the test data doesn't use the golden retrieve results, so the arguments in program may not
    #        occur in the arguments_list.
    # test_data_batch_generator = DataBatchGenerator(args, "test", OPERATION_LIST, CONST_LIST)

    train_data_batch_generator_finqa = DataBatchGenerator(args_2, "train", OPERATION_LIST_FINQA, CONST_LIST_FINQA)

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    ax = sns.histplot(data=train_data_batch_generator.operation_length, color="skyblue", label="MathQA", kde=True)
    ax = sns.histplot(data=train_data_batch_generator_finqa.operation_length, color="orange", label="FinQA", kde=True)
    ax.set(xlabel='Operation Length', ylabel='Count', title='Distribution Over Operation Length')
    plt.legend() 
    plt.show()
    # print(sum(train_data_batch_generator.operation_length) / len(train_data_batch_generator.operation_length))
    exit()

    for batch in train_data_batch_generator.data_loader(use_cuda=False):
        pass

    for batch in dev_data_batch_generator.data_loader(use_cuda=False):
        pass

    for batch in test_data_batch_generator.data_loader(use_cuda=False):
        pass


    # args.data_name = "drop"
    # args.cached_test_data = f"../datasets/cached_data/{args.data_name}/cached_dev_data.json"
    
    # test_drop_train_data_batch_generator = DataBatchGenerator(args, "test", OPERATION_LIST, CONST_LIST)

    # for batch in test_drop_train_data_batch_generator.data_loader(use_cuda=False):
    #     pass

    # args.data_name = "svamp"
    # args.cached_test_data = f"../datasets/cached_data/{args.data_name}/cached_test_data.json"
    
    # test_drop_train_data_batch_generator = DataBatchGenerator(args, "test", OPERATION_LIST, CONST_LIST)

    # for batch in test_drop_train_data_batch_generator.data_loader(use_cuda=False):
    #     pass