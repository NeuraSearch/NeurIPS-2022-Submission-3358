# coding:utf-8

import json
import torch
import codecs

from tqdm import tqdm
from argparse import ArgumentParser
from models.batch_generator import DataBatchGenerator
from models.finqa_model import FinQAModel
from models.model_wrapper import ModelWrapper
from utils import set_environment
from configurations import add_path_relevant_args, \
                            add_model_relevant_args, \
                                add_assist_args, \
                                    add_test_relevant_args

"""Parse the Arguments"""
parser = ArgumentParser("Test for FinQA.")
add_path_relevant_args(parser)
add_model_relevant_args(parser)
add_assist_args(parser)
add_test_relevant_args(parser)
args = parser.parse_args()
assert args.dropout_p == 0.0

data_name = args.data_name
if data_name in ["finqa", "drop", "svamp"]:
    from configurations import CONST_LIST as const_list
    from configurations import OPERATION_LIST as op_list
elif data_name == "mathqa":
    from configurations import CONST_LIST_MATHQA as const_list
    from configurations import OPERATION_LIST_MATHQA as op_list
else:
    raise NotImplementedError(f"Unknown data name: {data_name}")

"""Check GPU Available"""
args.cuda = torch.cuda.device_count() > 0
set_environment(args.seed, args.cuda)

"""Main function to execute test"""
def main():

    # create test data batch
    test_data_loder = DataBatchGenerator(args, "test", op_list=op_list, const_list=const_list).data_loader(use_cuda=args.cuda)

    # initialize the model
    model = FinQAModel(args=args, op_list=op_list, const_list=const_list)

    # wrap the model
    wrapped_model = ModelWrapper(
        args=args,
        model=model,
        mode="test"
    )

    print("start to test...")
    all_predictions = {}
    all_wrong_predictions = {}
    for batch_test in tqdm(test_data_loder):
        predictions, wrong_predictions = wrapped_model.evaluate(batch_test, op_list, const_list, False)
        all_predictions.update(predictions)
        all_wrong_predictions.update(wrong_predictions)
    
    exec_accu = wrapped_model.exec_accu.avg
    prog_accu = wrapped_model.prog_accu.avg
    wrapped_model.exec_accu.reset()
    wrapped_model.prog_accu.reset()

    print(f"Test {len(all_predictions)} datas Exec_accu: {exec_accu} Prog_accu: {prog_accu}")

    inference_results_path = args.inference_results_path
    with codecs.open(inference_results_path, "w") as file:
        json.dump(all_predictions, file, indent=2)
    
    inference_wrong_results_path = args.inference_wrong_results_path
    with codecs.open(inference_wrong_results_path, "w") as file:
        json.dump(all_wrong_predictions, file, indent=2)

    print(f"Finish Testing.")

if __name__ == "__main__":
    main()