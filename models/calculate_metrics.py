# coding:utf-8

# this file is used for calculating accuracies after prediction files been generated.
# Please NOTE: this is for MathQA only, for FinQA, please use "calculate_metrics_finqanet", which is official.

import copy
import json
import codecs
from model_wrapper import ModelWrapper

# customized here
# mathqa_prediction_file_path = "../prediction_results/1_test_results_mathqa_NrHd_RoBERTa-large.json"
mathqa_prediction_file_path = "../prediction_results/4_finqanet_mathqa_roberta-large.json"
mathqa_gold_file_path = "../datasets/cached_data/mathqa/cached_test_data.json"

def main(data_name="mathqa", program_is_sequence=False):
    if data_name == "mathqa":
        prediction_file_path = mathqa_prediction_file_path
        gold_file_path = mathqa_gold_file_path
    else:
        raise NotImplementedError

    total_count = 0
    prog_correct = 0
    op_correct = 0

    with codecs.open(prediction_file_path, "r", "utf-8") as file:
        predictions = json.load(file)
    
    with codecs.open(gold_file_path, "r", "utf-8") as file:
        golds = json.load(file)
    
    golds = {str(item["id"]): item for item in golds}

    if data_name == "mathqa":
        op_step_couts = {i: 0 for i in range(1, 17)}
        correct_couts = {i: 0 for i in range(1, 17)}
        correct_op_couts = {i: 0 for i in range(1, 17)}
    
    if data_name == "mathqa":
        cahce_num_counts = {i: 0 for i in range(-1, 15)}
        cache_num_correct = {i: 0 for i in range(-1, 15)}
    
    if not program_is_sequence:
        for data_id, entry in predictions.items():
            total_count +=1

            pred_op = entry["pred_op"]
            pred_argu = entry["pred_argu"]

            gold_data = golds[data_id]

            golden_op =gold_data["golden_op"]
            golden_argument = gold_data["golden_argument"]

            max_cache_num = -1
            for sub_argu_ids, sub_argu in enumerate(golden_argument):
                for argu in sub_argu:
                    if argu.startswith("#"):
                        cache_num = int(argu.replace("#", ""))
                        relative_cache_distance = sub_argu_ids - cache_num
                        if relative_cache_distance > max_cache_num:
                            max_cache_num = relative_cache_distance
            
            cahce_num_counts[max_cache_num] +=1

            invalid_flag = ModelWrapper.compare_program(
                operation=pred_op,
                arguments=pred_argu,
                golden_op=golden_op,
                golden_arguments=golden_argument,
                id=data_id,
            )

            if invalid_flag == 0:
                prog_correct +=1
                cache_num_correct[max_cache_num] +=1

            if data_name == "mathqa":
                op_flag = True
                for p_op, g_op in zip(pred_op, golden_op[:-1]):
                    if p_op != g_op:
                        op_flag = False
                        break
                if op_flag:
                    op_correct +=1

            # for different steps
            op_step = len(golden_op)
            op_step_couts[op_step - 1] +=1

            if data_name == "mathqa":
                if invalid_flag == 0:
                    correct_couts[op_step - 1] +=1
                if op_flag:
                    correct_op_couts[op_step - 1] +=1
        
    else:
        for _, entry in predictions.items():
            pred_program = entry["pred_program"][:-1]
            golden_program = entry["golden_program"][:-1]

            # get operation step
            op_step = golden_program.count(")")
            op_step_couts[op_step] += 1
            
            golden_op, golden_argu = extract_op_argu(golden_program)

            max_cache_num = -1
            for sub_argu_ids, sub_argu in enumerate(golden_argu):
                for argu in sub_argu:
                    if argu.startswith("#"):
                        cache_num = int(argu.replace("#", ""))
                        relative_cache_distance = sub_argu_ids - cache_num
                        if relative_cache_distance > max_cache_num:
                            max_cache_num = relative_cache_distance
            
            cahce_num_counts[max_cache_num] +=1

            golden_op.append("EOF")
            golden_argu.append(["none", "none"])
            pred_op, pred_argu = extract_op_argu(pred_program)

            invalid_flag = 1
            invalid_flag = ModelWrapper.compare_program(
                operation=pred_op,
                arguments=pred_argu,
                golden_op=golden_op,
                golden_arguments=golden_argu,
                id=-1,
            )

            if invalid_flag == 0:
                correct_couts[op_step] +=1

            if invalid_flag == 0:
                cache_num_correct[max_cache_num] +=1
            
            op_correct_flag = True
            for g_o, p_o in zip(golden_op, pred_op):
                if g_o != p_o:
                   op_correct_flag = False
                   break
            if op_correct_flag:
                correct_op_couts[op_step] += 1

    if data_name == "mathqa":
        if not program_is_sequence:
            print(f"Total Program Accuracy: {prog_correct / total_count * 100}")
            print(f"Total Operator Accuracy: {op_correct / total_count * 100}")
        bigger_than_3_correct = 0
        bigger_than_3_count = 0
        for op_step, total in op_step_couts.items():
            correct = correct_couts[op_step]
            correct_op = correct_op_couts[op_step]
            print(f"Step {op_step} Program Accuracy: {correct / total}")
            print(f"Step {op_step} Operator Accuracy: {correct_op / total}")

            if op_step >= 3:
                bigger_than_3_correct += correct
                bigger_than_3_count += total
        
        print(f"Larger or equal than 3 average accuracy: {bigger_than_3_correct / bigger_than_3_count}")

    for cache_num, cache_count in cahce_num_counts.items():
        if cache_num != -1 and cache_count != 0:
            correct = cache_num_correct[cache_num]
            print(f"Cache Number: {cache_num} Accuracy: {correct / cache_count}")


def extract_op_argu(program):
    op = []
    argu = []
    sub_argu = []
    prev_is_pare = False
    for token in program:
        if len(op) == 0:
            op.append(token)
        else:
            if prev_is_pare:
                argu.append(sub_argu)
                sub_argu = []
                op.append(token)
                prev_is_pare = False
            else:
                if token != ")":
                    sub_argu.append(token)
                else:
                    prev_is_pare = True

    argu.append(sub_argu)

    return op, argu

def convert_original_file_to_we_want(json_file):
    # This is used for coverting the original JSON predictions from the FinQANet,
    # to compare the program_steps, and maximum cache number distance.
    def str_to_num(text):

        text = text.replace(",", "")
        try:
            num = float(text)
        except ValueError:
            if "%" in text:
                text = text.replace("%", "")
                try:
                    num = float(text)
                    num = num / 100.0
                except ValueError:
                    num = "n/a"
            elif "const" in text:
                text = text.replace("const_", "")
                if text == "m1":
                    text = "-1"
                num = float(text)
            else:
                num = "n/a"
        return num
    
    def process_row(row_in):

        row_out = []
        invalid_flag = 0

        for num in row_in:
            num = num.replace("$", "").strip()
            num = num.split("(")[0].strip()

            num = str_to_num(num)

            if num == "n/a":
                invalid_flag = 1
                break

            row_out.append(num)

        if invalid_flag:
            return "n/a"

        return row_out

    with codecs.open(json_file, "r", "utf-8") as file:
        datas = json.load(file)

    final_data = {}
    for entry in datas:
        
        pred_prog = entry["qa"]["predicted"]
        program = "|".join(pred_prog)
        steps = program.split(")")[:-1]

        pred_op = []
        pred_argu = []
        res_dict = {}
        for ind, step in enumerate(steps):
            step = step.strip()

            if len(step.split("(")) != 2:
                pred_op.append("none")
                pred_argu.append(["none", "none"])
                continue

            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            if len(args.split("|")) > 1:
                arg1 = args.split("|")[0].strip()
                arg2 = args.split("|")[1].strip()
            else:
                arg1 = args.split("|")[0].strip()
                arg2 = "none"

            pred_op.append(op)
            pred_argu.append([arg1, arg2])

        this_res = "n/a"
        table = entry["table"]
        try:
            for ind, step in enumerate(steps):
                step = step.strip()

                op = step.split("(")[0].strip("|").strip()
                args = step.split("(")[1].strip("|").strip()

                arg1 = args.split("|")[0].strip()
                arg2 = args.split("|")[1].strip()

                if op == "add" or op == "subtract" or op == "multiply" or op == "divide" or op == "exp" or op == "greater":

                    if "#" in arg1:
                        arg1 = res_dict[int(arg1.replace("#", ""))]
                    else:
                        arg1 = str_to_num(arg1)
                        if arg1 == "n/a":
                            invalid_flag = 1
                            break

                    if "#" in arg2:
                        arg2 = res_dict[int(arg2.replace("#", ""))]
                    else:
                        arg2 = str_to_num(arg2)
                        if arg2 == "n/a":
                            invalid_flag = 1
                            break

                    if op == "add":
                        this_res = arg1 + arg2
                    elif op == "subtract":
                        this_res = arg1 - arg2
                    elif op == "multiply":
                        this_res = arg1 * arg2
                    elif op == "divide":
                        this_res = arg1 / arg2
                    elif op == "exp":
                        this_res = arg1 ** arg2
                    elif op == "greater":
                        this_res = "yes" if arg1 > arg2 else "no"

                    res_dict[ind] = this_res

                elif "table" in op:
                    table_dict = {}
                    for row in table:
                        table_dict[row[0]] = row[1:]

                    if "#" in arg1:
                        arg1 = res_dict[int(arg1.replace("#", ""))]
                    else:
                        if arg1 not in table_dict:
                            invalid_flag = 1
                            break

                        cal_row = table_dict[arg1]
                        num_row = process_row(cal_row)

                    if num_row == "n/a":
                        invalid_flag = 1
                        break
                    if op == "table_max":
                        this_res = max(num_row)
                    elif op == "table_min":
                        this_res = min(num_row)
                    elif op == "table_sum":
                        this_res = sum(num_row)
                    elif op == "table_average":
                        this_res = sum(num_row) / len(num_row)

                    res_dict[ind] = this_res

        except:
            pass

        if this_res != "yes" and this_res != "no" and this_res != "n/a":

            this_res = round(this_res, 5)

        final_data[entry["id"]] = {
            "pred_op": pred_op,
            "pred_argu": pred_argu,
            "pred_exec_res": this_res,
            "golden_prog": entry["qa"]["program"],
            "golden_exe_ans": entry["qa"]["exe_ans"]
        }

    return final_data

if __name__ == "__main__":

    # main(data_name="mathqa")

    main(data_name="mathqa", program_is_sequence=True)
