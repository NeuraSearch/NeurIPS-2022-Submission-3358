# coding:utf-8
# This is from the original FinQANet code,
# There is a few revisements for the code, because of our JSON file formats are different,
# However, no touch to the algorithm.

import json
from sympy import simplify

all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
        "table_min", "table_sum", "table_average"]

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

def eval_program(program, table):
    '''
    calculate the numerical results of the program
    '''

    invalid_flag = 0
    this_res = "n/a"

    try:
        program = program[:-1]  # remove EOF
        # check structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"

        program = "|".join(program)
        steps = program.split(")")[:-1]
        res_dict = {}

        for ind, step in enumerate(steps):
            step = step.strip()

            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
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
        if this_res != "yes" and this_res != "no" and this_res != "n/a":

            this_res = round(this_res, 5)

    except:
        invalid_flag = 1

    return invalid_flag, this_res

def equal_program(program1, program2):
    '''
    symbolic program if equal
    program1: gold
    program2: pred
    '''

    sym_map = {}

    program1 = program1[:-1]  # remove EOF
    program1 = "|".join(program1)
    steps = program1.split(")")[:-1]

    invalid_flag = 0
    sym_ind = 0
    step_dict_1 = {}


    # symbolic map
    for ind, step in enumerate(steps):

        step = step.strip()

        assert len(step.split("(")) <= 2

        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()

        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        step_dict_1[ind] = step

        if "table" in op:
            if step not in sym_map:
                sym_map[step] = "a" + str(sym_ind)
                sym_ind += 1

        else:
            if "#" not in arg1:
                if arg1 not in sym_map:
                    sym_map[arg1] = "a" + str(sym_ind)
                    sym_ind += 1

            if "#" not in arg2:
                if arg2 not in sym_map:
                    sym_map[arg2] = "a" + str(sym_ind)
                    sym_ind += 1

    # check program 2
    step_dict_2 = {}
    try:
        program2 = program2[:-1]  # remove EOF
        # check structure
        for ind, token in enumerate(program2):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    print("structure error")
                    return False
            if (ind + 1) % 4 == 0:
                if token != ")":
                    print("structure error")
                    return False

        program2 = "|".join(program2)
        steps = program2.split(")")[:-1]


        for ind, step in enumerate(steps):
            step = step.strip()

            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            step_dict_2[ind] = step

            if "table" in op:
                if step not in sym_map:
                    return False

            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        return False
                else:
                    if int(arg1.strip("#")) >= ind:
                        return False

                if "#" not in arg2:
                    if arg2 not in sym_map:
                        return False
                else:
                    if int(arg2.strip("#")) >= ind:
                        return False
    except:
        return False

    def symbol_recur(step, step_dict):

        step = step.strip()
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()

        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        if "table" in op:
            # as var
            return sym_map[step]

        if "#" in arg1:
            arg1_ind = int(arg1.replace("#", ""))
            arg1_part = symbol_recur(step_dict[arg1_ind], step_dict)
        else:
            arg1_part = sym_map[arg1]

        if "#" in arg2:
            arg2_ind = int(arg2.replace("#", ""))
            arg2_part = symbol_recur(step_dict[arg2_ind], step_dict)
        else:
            arg2_part = sym_map[arg2]

        if op == "add":
            return "( " + arg1_part + " + " + arg2_part + " )"
        elif op == "subtract":
            return "( " + arg1_part + " - " + arg2_part + " )"
        elif op == "multiply":
            return "( " + arg1_part + " * " + arg2_part + " )"
        elif op == "divide":
            return "( " + arg1_part + " / " + arg2_part + " )"
        elif op == "exp":
            return "( " + arg1_part + " ** " + arg2_part + " )"
        elif op == "greater":
            return "( " + arg1_part + " > " + arg2_part + " )"

    # # derive symbolic program 1
    steps = program1.split(")")[:-1]
    sym_prog1 = symbol_recur(steps[-1], step_dict_1)
    sym_prog1 = simplify(sym_prog1, evaluate=False)

    try:
        # derive symbolic program 2
        steps = program2.split(")")[:-1]
        sym_prog2 = symbol_recur(steps[-1], step_dict_2)
        sym_prog2 = simplify(sym_prog2, evaluate=False)
    except:
        return False

    return sym_prog1 == sym_prog2

def evaluate_result(json_in, json_ori, our_gold_json):
    '''
    execution acc
    program acc
    '''
    correct = 0

    with open(json_in) as f_in:
        data = json.load(f_in)

    with open(json_ori) as f_in:
        data_ori = json.load(f_in)

    with open(our_gold_json) as f_our:
        data_our = json.load(f_our)

    data_dict = {}
    for each_data in data_ori:
        assert each_data["id"] not in data_dict
        data_dict[each_data["id"]] = each_data

    our_data_dict = {}
    for each_data in data_our:
        assert each_data["id"] not in our_data_dict
        our_data_dict[each_data["id"]] = each_data

    exe_correct = 0
    prog_correct = 0

    res_list = []
    all_res_list = []

    # NOTE: we change here, since the prediction file keys are different.
    # NOTE: we need calculate the program accuracy
    op_step_couts = {i: 0 for i in range(1, 4)}
    correct_couts = {i: 0 for i in range(1, 4)}
    exe_correct_counts = {i:0 for i in range(1, 4)}
    cahce_num_counts = {i: 0 for i in range(-1, 11)}
    cache_num_correct = {i: 0 for i in range(-1, 11)}
    for each_id, each_data in data.items():

        each_ori_data = data_dict[each_id]

        table = each_ori_data["table"]
        gold_res = each_ori_data["qa"]["exe_ans"]

        # NOTE: we change the key names here, since the JSON save is different.
        pred_op = each_data["pred_op"]
        pred_argu = each_data["pred_argu"]
        # gold = each_data["golden_prog"]
        gold_op = our_data_dict[each_id]["golden_op"]
        gold_argu = our_data_dict[each_id]["golden_argument"]

        """
        In order to use their evaluation metrics, we need to format our prediction to their style:
            "predicted": [
            "subtract(",
            "5829",
            "5735",
            ")",
            "EOF"
            ]
        However, since this is hard code because of one operator only supports exact two operands,
        The evaluation method is limited.
        """
        pred = []
        for op, argu in zip(pred_op, pred_argu):
            temp = []
            temp.append(f"{op}(")
            temp.extend([str(_) for _ in argu])
            if len(temp) == 2:
                temp.append("none")
            temp.append(")")
            pred.extend(temp)
        pred.append("EOF")

        gold = []
        for op, argu in zip(gold_op[:-1], gold_argu[:-1]):
            gold.append(f"{op}(")
            gold.extend([_ for _ in argu])
            gold.append(")")
        gold.append("EOF")

        if len(gold_op[:-1]) in op_step_couts:
            op_step_couts[len(gold_op[:-1])] +=1
        else:
            op_step_couts[3] += 1

        max_cache_num = -1
        for sub_argu_ids, sub_argu in enumerate(gold_argu):
            for argu in sub_argu:
                if argu.startswith("#"):
                    cache_num = int(argu.replace("#", ""))
                    relative_cache_distance = sub_argu_ids - cache_num
                    if relative_cache_distance > max_cache_num:
                        max_cache_num = relative_cache_distance

        cahce_num_counts[max_cache_num] +=1

        invalid_flag, exe_res = eval_program(pred, table)

        exec_equal = False
        if invalid_flag == 0:
            if exe_res == gold_res:
                exe_correct += 1
                exec_equal = True
            try:
                if float(exe_res) / 100 == float(gold_res) or float(gold_res) / 100 == float(exe_res):
                    exe_correct += 1
                    exec_equal = True
            except ValueError:
                pass

        if exec_equal:
            if len(gold_op[:-1]) in exe_correct_counts:
                exe_correct_counts[len(gold_op[:-1])] +=1
            else:
                exe_correct_counts[3] += 1        


        if equal_program(gold, pred):
            if exe_res != gold_res:
                exit()
                print(each_id)
                print(gold)
                print(pred)
                print(gold_res)
                print(exe_res)
                print(each_ori_data["id"])
            assert exe_res == gold_res
            prog_correct += 1
            if "".join(gold) != "".join(pred):
                print(each_id)
                print(gold)
                print(pred)
                print(gold_res)
                print(exe_res)
                print(each_ori_data["id"])

            if len(gold_op[:-1]) in correct_couts:
                correct_couts[len(gold_op[:-1])] +=1
            else:
                correct_couts[3] += 1
            cache_num_correct[max_cache_num] +=1

        each_ori_data["qa"]["predicted"] = pred

        if exe_res != gold_res:
            res_list.append(each_ori_data)
        all_res_list.append(each_ori_data)

    exe_acc = float(exe_correct) / len(data)
    prog_acc = float(prog_correct) / len(data)

    print("All: ", len(data))
    print("Correct: ", correct)
    print("Exe acc: ", exe_acc)
    print("Prog acc: ", prog_acc)

    for op_step in op_step_couts:
        cur_op_step_correct = correct_couts[op_step]
        cur_op_step_total = op_step_couts[op_step]
        cur_op_exec_correct = exe_correct_counts[op_step]
        print(f"OP_STEP: {op_step} Accuracy: {cur_op_step_correct / cur_op_step_total}")
        print(f"OP_STEP: {op_step} Exec_Accuracy: {cur_op_exec_correct / cur_op_step_total}")

    for cur_max_cache_num, cur_max_cache_count in cahce_num_counts.items():
        if cur_max_cache_num != -1 and cur_max_cache_count != 0:
            print(f"MAX_CACHE_NUM: {cur_max_cache_num} Accuracy: {cache_num_correct[cur_max_cache_num] / cur_max_cache_count}")

if __name__ == "__main__":
    evaluate_result(
        json_in="../prediction_results/3_test_results_finqa_NrHd_RoBERTa-large.json",
        json_ori="../datasets/raw_data/finqa/test_retrieve.json",
        our_gold_json="../datasets/cached_data/finqa/cached_test_data.json"
    ) 