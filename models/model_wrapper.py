# coding:utf-8

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import copy
import functools
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import get_linear_schedule_with_warmup
from utils import AverageMeter, str_to_num_2, inverse_sigmoid_decay
from models.calculate_metrics_finqanet import eval_program, equal_program

class ModelWrapper:
    def __init__(self, *args, **kwargs):
        if kwargs["mode"] in ["train", "eval"]:
            self.__init_train__(
                args=kwargs["args"],
                model=kwargs["model"],
                num_train_steps=kwargs["num_train_steps"],
                logger=kwargs["logger"],
                is_program_as_sequence=kwargs["is_program_as_sequence"]
            )
        elif kwargs["mode"] == "test":
            self.__init_test__(
                args=kwargs["args"],
                model=kwargs["model"],
            )

    def __init_train__(self, args, model, num_train_steps, logger, is_program_as_sequence):
        self.args = copy.deepcopy(args)
        self.num_train_steps = num_train_steps
        self.model = model
        self.logger = logger
        self.data_name = args.data_name

        self.epoch = 1
        # self.step = self.updates * gradient_accumulation_steps
        self.step = 0
        self.updates = 0
        if self.args.sheduled_sampling == 1:
            self.sampling_fn = functools.partial(inverse_sigmoid_decay, k=self.args.sampling_k)
        else:
            self.sampling_fn = lambda i: 1.0

        # record the metrics 
        self.train_loss = AverageMeter()
        self.eval_loss = AverageMeter()
        self.op_loss = AverageMeter()
        self.argu_loss = AverageMeter()
        self.prog_accu = AverageMeter()
        self.exec_accu = AverageMeter()

        # check whether reloading the model from the previous checkpoint
        self.restore_from_prev = False
        if self.args.reload_model_path:
            self.restore_from_prev = True

            self.logger.info("restore model...")
            self.model.load_state_dict(torch.load(args.reload_model_path))
            self.epoch = torch.load(args.reload_config_path)["epoch"]
            self.step += torch.load(args.reload_config_path)["step"]
            self.last_best_prog_accu = torch.load(args.reload_config_path)["eval_score"]

        # set the updated parameters
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        if self.args.fine_tune:
            logger.info("Fine tune PLM model.")
            if not is_program_as_sequence:
                optimized_parameters = [
                    {"params": [p for n, p in self.model.plm_model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay, "lr": self.args.lr},
                    {"params": [p for n, p in self.model.plm_model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0, "lr": self.args.lr},
                    {"params": [p for n, p in self.model.named_parameters() if (not n.startswith("plm_model.")) and (not any(nd in n for nd in no_decay)) and p.requires_grad],
                    "weight_decay": self.args.weight_decay, "lr": self.args.lr},
                    {"params": [p for n, p in self.model.named_parameters() if (not n.startswith("plm_model.")) and (any(nd in n for nd in no_decay)) and p.requires_grad],
                    "weight_decay": 0.0, "lr": self.args.lr},
                ]
            else:
                optimized_parameters = [
                    {"params": [p for n, p in self.model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay, "lr": self.args.lr},
                    {"params": [p for n, p in self.model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0, "lr": self.args.lr},
                    {"params": [p for n, p in self.model.named_parameters() if (not n.startswith("bert.")) and (not any(nd in n for nd in no_decay)) and p.requires_grad],
                    "weight_decay": self.args.weight_decay, "lr": self.args.lr},
                    {"params": [p for n, p in self.model.named_parameters() if (not n.startswith("bert.")) and (any(nd in n for nd in no_decay)) and p.requires_grad],
                    "weight_decay": 0.0, "lr": self.args.lr},
                ]

                self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        else:
            logger.info("Fixed the PLM model.")
            optimized_parameters = [
                {"params": [p for n, p in self.model.named_parameters() if not n.startswith("plm_model.")],
                "weight_decay": self.args.weight_decay, "lr": self.args.lr},
            ]

        self.total_param = sum([p.nelement() for p in self.model.parameters() if p.requires_grad])
        self.logger.info(f"Total Parameters to update: {self.total_param}")

        # check whether train on GPU
        if self.args.cuda:
            self.model.cuda()
        
        # set optimizer and learning scheduler
        self.optimizer = optim.Adam(params=optimized_parameters)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0,
            num_training_steps=self.num_train_steps,
        )
        if self.restore_from_prev:
            self.optimizer.load_state_dict(torch.load(args.reload_optimizer_path))
            self.lr_scheduler.load_state_dict(torch.load(args.reload_scheduler_path))
        self.optimizer.zero_grad()
    
    def __init_test__(self, args, model):
        self.args = copy.deepcopy(args)
        self.model = model
        self.data_name = args.data_name

        self.prog_accu = AverageMeter()
        self.exec_accu = AverageMeter()

        assert self.args.reload_model_path is not None
        self.model.load_state_dict(torch.load(self.args.reload_model_path))

        if self.args.cuda:
            self.model.cuda()
    
    def update(self, batch):
        self.model.train()

        # get the batch inputs for the model
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]
        input_question_mask = batch["input_question_mask"]
        input_number_mask = batch["input_number_mask"]
        golden_op = batch["golden_op"]
        golden_op_mask = batch["golden_op_mask"]
        golden_arguments = batch["golden_arguments"]
        golden_argu_mask = batch["golden_argu_mask"]

        input_number_unit_graph = batch["input_number_unit_graph"]
        input_number_number_graph = batch["input_number_number_graph"]
        input_number_indices = batch["input_number_indices"]
        graph_number_mask = batch["graph_number_mask"]
        input_unit_ids = batch["input_unit_ids"]

        sampling_rate = 1 - self.sampling_fn(i=self.step)

        # forward
        outputs = self.model(
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            input_number_unit_graph=input_number_unit_graph,
            input_number_number_graph=input_number_number_graph,
            input_number_indices=input_number_indices,
            graph_number_mask=graph_number_mask,
            input_unit_ids=input_unit_ids,
            input_question_mask=input_question_mask,
            input_number_mask=input_number_mask,
            golden_op=golden_op,
            golden_op_mask=golden_op_mask,
            golden_arguments=golden_arguments,
            golden_argu_mask=golden_argu_mask,
            sampling_rate=sampling_rate
        )

        # get the loss
        op_loss = outputs.op_loss
        argu_loss = outputs.argu_loss
        loss = op_loss + argu_loss
        op_loss_scalar = op_loss.clone().detach().item()
        argu_loss_scalar = argu_loss.clone().detach().item()
        loss_scalar = loss.clone().detach().item()
        self.op_loss.update(op_loss_scalar, 1)
        self.argu_loss.update(argu_loss_scalar, 1)
        self.train_loss.update(loss_scalar, 1)

        # calculate the gradients
        if self.args.gradient_accumulation_steps > 1:
            loss /= self.args.gradient_accumulation_steps
        loss.backward()

        # update the parameters
        if self.step % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.updates +=1
        self.step +=1
        self.lr = self.lr_scheduler.get_last_lr()[-1]
    
    def update_finqa_baseline(self, batch, op_list, const_list):
        self.model.train()
        # is_training, input_ids, input_mask, segment_ids, program_ids, option_mask

        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]

        program_ids = batch["golden_program"]
        program_mask = batch["golden_program_mask"]

        option_list = op_list + const_list
        option_mask = torch.LongTensor([1 if opt != "GO" else 0 for opt in option_list]).to(input_ids.device)
        option_mask = torch.unsqueeze(option_mask, 0).repeat(input_ids.size(0), 1)
        input_number_mask = batch["input_number_mask"]

        option_num_mask = torch.cat((option_mask, input_number_mask), dim=1)
        
        logits, _ = self.model(
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            program_ids=program_ids,
            option_mask=option_num_mask,
        )

        bsz, prog_len, _ = logits.size()

        golden_program_flatten = torch.reshape(program_ids, (-1, ))
        logits_reshape = torch.reshape(logits, (bsz * prog_len, -1))
        program_mask_flatten = torch.reshape(program_mask, (-1, ))

        loss = self.loss_fn(input=logits_reshape, target=golden_program_flatten)
        loss = torch.sum(loss * program_mask_flatten) / torch.sum(program_mask_flatten)

        loss_scalar = loss.item()

        self.train_loss.update(loss_scalar, 1)

        # calculate the gradients
        if self.args.gradient_accumulation_steps > 1:
            loss /= self.args.gradient_accumulation_steps
        loss.backward()

        # update the parameters
        if self.step % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.updates +=1
        self.step +=1
        self.lr = self.lr_scheduler.get_last_lr()[-1]

    @torch.no_grad()
    def evaluate(self, batch, op_list, const_list, require_loss):
        self.model.eval()

        # get the batch inputs for the model
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]
        input_question_mask = batch["input_question_mask"]
        input_number_mask = batch["input_number_mask"]
        input_number_unit_graph = batch["input_number_unit_graph"]
        input_number_number_graph = batch["input_number_number_graph"]
        input_number_indices = batch["input_number_indices"]
        graph_number_mask = batch["graph_number_mask"]
        input_unit_ids = batch["input_unit_ids"]
        if require_loss:
            golden_op = batch["golden_op"]
            golden_op_mask = batch["golden_op_mask"]
            golden_arguments = batch["golden_arguments"]
            golden_argu_mask = batch["golden_argu_mask"]
        else:
            golden_op = None
            golden_op_mask = None
            golden_arguments = None
            golden_argu_mask = None

        # get the golden data for evaluation
        instance_set = batch["metadata"]
        data_id = [instance["id"] for instance in instance_set]
        if self.data_name == "finqa":
            tables = [instance["tables"] for instance in instance_set]
            golden_exe_ans = [instance["original_exe_ans"] for instance in instance_set]
        elif self.data_name == "svamp":
            golden_exe_ans = [instance["original_exe_ans"] for instance in instance_set]
        elif self.data_name == "drop_annotated":
            golden_exe_ans = [instance["golden_answer"] for instance in instance_set]
        elif self.data_name in ["drop_fewshot", "drop_fakedata"]:
            golden_exe_ans = [instance["original_exe_ans"] for instance in instance_set]

        arguments = [instance["arguments"] for instance in instance_set]
        argument_indices_offset = [instance["argument_indices_offset"] for instance in instance_set]

        if self.data_name in ["finqa", "mathqa", "svamp"]:
            original_program = [instance["original_program"] for instance in instance_set]

        if self.data_name == "drop":
            golden_exe_ans = [instance["original_exe_ans"] for instance in instance_set]
        
        if self.data_name in ["drop_annotated", "drop_fakedata"]:
            original_program = [[] for instance in instance_set]

        # forward
        outputs = self.model(
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            input_number_unit_graph=input_number_unit_graph,
            input_number_number_graph=input_number_number_graph,
            input_number_indices=input_number_indices,
            graph_number_mask=graph_number_mask,
            input_unit_ids=input_unit_ids,
            input_question_mask=input_question_mask,
            input_number_mask=input_number_mask,
            golden_op=golden_op,
            golden_op_mask=golden_op_mask,
            golden_arguments=golden_arguments,
            golden_argu_mask=golden_argu_mask,
            requires_loss=require_loss,
        )

        # get the outputs
        # [bsz, max_op]
        operation_ids = outputs.operation_ids
        # [bsz, max_op, max_argu]
        arguments_ids = outputs.arguments_ids
        if require_loss:
            op_loss_scalar = outputs.op_loss.clone().detach().item()
            argu_loss_scalar = outputs.argu_loss.clone().detach().item()
            loss_scalar = op_loss_scalar + argu_loss_scalar
            self.eval_loss.update(loss_scalar, 1)

        # op_ids -> op
        # actual_op_len is the actual length for each data in the batch,
        # where the length refers to the number of op before "EOF".
        predicted_op, actual_op_len = self.op_ids_to_op(operation_ids, op_list)

        # argu_ids -> argu
        if self.data_name in ["finqa", "drop", "svamp", "drop_annotated", "drop_fewshot", "drop_fakedata"]:
            predicted_argu = self.argu_ids_to_argu(
                arguments_ids=arguments_ids,
                const_list=const_list,
                arguments=arguments,
                argument_indices_offset=argument_indices_offset,
                actual_op_len=actual_op_len,
            )
        elif self.data_name in ["mathqa"]:
            predicted_argu = self.argu_ids_to_argu_mathqa(
                arguments_ids=arguments_ids,
                const_list=const_list,
                argument_indices_offset=argument_indices_offset,
                actual_op_len=actual_op_len,
            )

        predictions = {}
        wrong_predictions = {}
        # NOTE: use e_bsz may cause IndexError, Since one batch may not be full.
        for i in range(len(predicted_op)):
            pred_operation=predicted_op[i]
            pred_arguments=predicted_argu[i]

            if self.data_name in ["finqa", "mathqa", "svamp", "drop_annotated"]:
                golden_prog = original_program[i]

                golden_op = instance_set[i]["golden_op"]
                golden_arguments = instance_set[i]["golden_argument"]
            elif self.data_name in ["drop", "drop_fewshot", "drop_fakedata"]:
                golden_prog = []

                golden_op = []
                golden_arguments = []

            if self.data_name in ["drop", "svamp", "drop_annotated", "drop_fewshot", "drop_fakedata"]:
                # execute result accuracy
                invalid_flag, pred_exec_res = self.calculate_program(
                    operation=pred_operation,
                    arguments=pred_arguments,
                    table=tables[i] if self.data_name == "finqa" else None,
                )

                # compare the execute accuracy
                execu_correct_number = 0
                if invalid_flag == 0:
                    # compare when both are numbers
                    if type(pred_exec_res) in [int, float] and type(golden_exe_ans[i]) in [int, float]:
                        if pred_exec_res == golden_exe_ans[i] or (abs(pred_exec_res - golden_exe_ans[i]) < 1e-2):
                            execu_correct_number = 1
                    # compare when at least one is str
                    elif type(pred_exec_res) == str or type(golden_exe_ans[i]) == str:
                        if type(pred_exec_res) == str and type(golden_exe_ans[i]) == str:
                            if pred_exec_res == golden_exe_ans[i]:
                                execu_correct_number = 1
                        elif type(pred_exec_res) == str and (type(golden_exe_ans[i]) != str):
                            if str_to_num_2(pred_exec_res, convert_const=True) == golden_exe_ans[i]:
                                execu_correct_number = 1
                        elif type(golden_exe_ans[i]) == str and (type(pred_exec_res) != str):
                            if str_to_num_2(golden_exe_ans[i], convert_const=True) == pred_exec_res:
                                execu_correct_number = 1
                self.exec_accu.update(execu_correct_number, 1)
            elif self.data_name == "finqa":
                pred_finqa_prog = []
                for op, argu in zip(pred_operation, pred_arguments):
                    temp = []
                    temp.append(f"{op}(")
                    temp.extend([str(_) for _ in argu])
                    if len(temp) == 2:
                        temp.append("none")
                    temp.append(")")
                    pred_finqa_prog.extend(temp)
                pred_finqa_prog.append("EOF")

                gold_finqa_prog = []
                for op, argu in zip(golden_op[:-1], golden_arguments[:-1]):
                    gold_finqa_prog.append(f"{op}(")
                    gold_finqa_prog.extend([_ for _ in argu])
                    gold_finqa_prog.append(")")
                gold_finqa_prog.append("EOF")

                invalid_flag, exe_res = eval_program(pred_finqa_prog, tables[i])
                pred_exec_res = exe_res
                
                execu_correct_number = 0
                if invalid_flag == 0:
                    if exe_res == golden_exe_ans[i]:
                       self.exec_accu.update(1, 1)
                       execu_correct_number = 1
                    try:
                        if float(exe_res) / 100 == float(golden_exe_ans[i]) or float(golden_exe_ans[i]) / 100 == float(exe_res):
                            self.exec_accu.update(1, 1)
                            execu_correct_number = 1
                    except ValueError:
                        pass
                    if execu_correct_number == 0:
                        self.exec_accu.update(0, 1)
                else:
                    self.exec_accu.update(0, 1)
                
                invalid_flag_prog = 1
                if equal_program(gold_finqa_prog, pred_finqa_prog):
                    assert exe_res == golden_exe_ans[i]
                    # self.prog_accu.update(1, 1)
                    invalid_flag_prog = 0
            else:
                pred_exec_res = 0
                execu_correct_number = 0
                self.exec_accu.update(0, 1)


            if self.data_name in ["mathqa", "drop_annotated"]:
                invalid_flag_prog = self.compare_program_mathqa(
                    operation=pred_operation,
                    arguments=pred_arguments,
                    golden_op=golden_op,
                    golden_arguments=golden_arguments,               
                )
            elif self.data_name in ["drop", "drop_fewshot", "drop_fakedata"]:
                invalid_flag_prog = 1

            prog_correct_number = 0
            if invalid_flag_prog == 0:
                prog_correct_number = 1

            # save predictions
            predictions[data_id[i]] = {
                "pred_op": pred_operation,
                "pred_argu": pred_arguments,
                "pred_exec_res": pred_exec_res,
                "golden_prog": golden_prog,
                "golden_exe_ans": golden_exe_ans[i] if self.data_name in ["finqa", "drop", "svamp", "drop_annotated", "drop_fewshot", "drop_fakedata"] else None
            }
            
            # sometimes the execution numbers are the same, however the programs are not,
            # this may contains the programs which are actually same, just small difference.
            if self.data_name in ["mathqa", "svamp", "drop_annotated"]:
                if execu_correct_number == 1 and prog_correct_number == 0:
                    revise_flag = 0
                    if len(pred_arguments) == len(golden_arguments[:-1]):
                        for p_argu, g_argu in zip(pred_arguments, golden_arguments):
                            for p, g in zip(p_argu, g_argu):
                                if ("#" not in g) and ("none" not in g):
                                    if g.startswith("-"):
                                        revise_flag = 1
                                    elif "%" in g:
                                        revise_flag = 1
                                    elif (type(p) in [int, float]) and (p > 0):
                                        revise_flag = 1
                    if revise_flag == 1:
                        prog_correct_number = 1
                    else:
                        # for the "table" operation, some will use table operation with row[0] as argument,
                        # some will use the numbers of that row. e.g., table_average(row[0]) and sum(row[1], row[2]), divide(#1, 2).
                        for op in pred_operation:
                            if "table" in op:
                                prog_correct_number = 1
                        for op in golden_op:
                            if "table" in op:
                                prog_correct_number = 1

            self.prog_accu.update(prog_correct_number, 1)

            if self.data_name in ["finqa", "drop_annotated"]:
                if execu_correct_number == 0 or prog_correct_number == 0:
                    wrong_predictions[data_id[i]] = {
                        "pred_op": pred_operation,
                        "pred_argu": pred_arguments,
                        "pred_exec_res": pred_exec_res,
                        "golden_prog": golden_prog,
                        "golden_exe_ans": golden_exe_ans[i],
                        "golden_op": golden_op,
                        "golden_arguments": golden_arguments,
                    }
            elif self.data_name == "mathqa":
                if prog_correct_number == 0:
                    wrong_predictions[data_id[i]] = {
                        "pred_op": pred_operation,
                        "pred_argu": pred_arguments,
                        "pred_exec_res": pred_exec_res,
                        "golden_prog": golden_prog,
                        "golden_exe_ans": None,
                    } 
            elif self.data_name in ["drop", "drop_fewshot", "drop_fakedata"]:
                if execu_correct_number == 0:
                    wrong_predictions[data_id[i]] = {
                        "pred_op": pred_operation,
                        "pred_argu": pred_arguments,
                        "pred_exec_res": pred_exec_res,
                        "golden_exe_ans": golden_exe_ans[i],
                    }                   
            elif self.data_name == "svamp":
                if prog_correct_number == 0 or execu_correct_number == 0:
                    wrong_predictions[data_id[i]] = {
                        "pred_op": pred_operation,
                        "pred_argu": pred_arguments,
                        "pred_exec_res": pred_exec_res,
                        "golden_prog": golden_prog,
                        "golden_exe_ans": golden_exe_ans[i],
                    } 

        # only evaluation after training requires loss
        if require_loss:
            self.model.train()

        return predictions, wrong_predictions

    @torch.no_grad()
    def evaluate_finqa_baseline(self, batch, op_list, const_list):
        self.model.eval()
        # is_training, input_ids, input_mask, segment_ids, program_ids, option_mask

        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]

        program_ids = batch["golden_program"]

        instance_set = batch["metadata"]
        argument_indices_offset = [instance["argument_indices_offset"] for instance in instance_set]
        golden_sequence_program = [instance["sequence_program"] for instance in instance_set]
        data_id = [instance["id"] for instance in instance_set]

        option_list = op_list + const_list
        option_mask = torch.LongTensor([1 if opt != "GO" else 0 for opt in option_list]).to(input_ids.device)
        option_mask = torch.unsqueeze(option_mask, 0).repeat(input_ids.size(0), 1)
        input_number_mask = batch["input_number_mask"]

        option_num_mask = torch.concat((option_mask, input_number_mask), dim=1)

        _, prediction_ids = self.model(
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            program_ids=program_ids,
            option_mask=option_num_mask,
        )

        prediction_ids = prediction_ids.tolist()

        prediction_program_token = self.program_ids_to_token(
            program_ids=prediction_ids,
            op_list=op_list,
            const_list=const_list,
            argument_indices_offset=argument_indices_offset,
        )

        invalid_flag = 0
        batch_predictions = {}
        wrong_predictions = {}
        for i, (gold, pred) in enumerate(zip(golden_sequence_program, prediction_program_token)):
            gold = gold[:-1]
            if len(gold) != len(pred):
                invalid_flag = 1
            else:
                sub_gold_cache = []
                sub_pred_cache = []
                for t_g, t_p in zip(gold, pred):
                    if t_g == ")":
                        if t_p != ")":
                            invalid_flag = 1
                            break
                        else:
                            for sub_gold_t in sub_gold_cache:
                                if sub_gold_t not in sub_pred_cache:
                                    invalid_flag = 1
                                    break
                        sub_gold_cache = []
                        sub_pred_cache = []
                    else:
                        sub_gold_cache.append(t_g)
                        sub_pred_cache.append(t_p)

            batch_predictions[data_id[i]] = {
                "pred_program": pred,
                "golden_program": gold,
            }
            if invalid_flag == 1:
                wrong_predictions[data_id[i]] = {
                    "pred_program": pred,
                    "golden_program": gold,
                }
            self.prog_accu.update(1 - invalid_flag, 1)
            invalid_flag = 0

        return batch_predictions, wrong_predictions
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    def avg_reset(self):
        self.train_loss.reset()
        self.eval_loss.reset()
        self.op_loss.reset()
        self.argu_loss.reset()
        self.prog_accu.reset()
        self.exec_accu.reset()
    
    def save(self, prefix, epoch, eval_score=None):
        model_state = dict([(k, v.cpu()) for k, v in self.model.state_dict().items()])
        other_params = {
            "epoch": epoch,
            "step": self.step,
            "eval_score": eval_score if eval_score is not None else -1,
        }
        optimizer_params = self.optimizer.state_dict()
        lr_scheduler_params = self.lr_scheduler.state_dict()

        model_save_path = prefix + ".pt"
        other_params_path = prefix + ".ct"
        optimizer_save_path = prefix + ".op"
        lr_scheduler_path = prefix + ".lr"

        torch.save(model_state, model_save_path)
        torch.save(other_params, other_params_path)
        torch.save(optimizer_params, optimizer_save_path)
        torch.save(lr_scheduler_params, lr_scheduler_path)

        self.logger.info(f"model and configurations saved to {prefix}.")

    @staticmethod    
    def op_ids_to_op(operation_ids, op_list):
        operation_ids_list = operation_ids.tolist()

        predicted_op = []
        for batch_op_ids in operation_ids_list:
            batch_op = []
            for op_ids in batch_op_ids:
                cur_op = op_list[op_ids]
                if cur_op == "EOF":
                    break
                batch_op.append(op_list[op_ids])
            predicted_op.append(batch_op)
        
        actual_op_len = [len(batch_op) for batch_op in predicted_op]

        return predicted_op, actual_op_len
    
    @staticmethod
    def argu_ids_to_argu(arguments_ids, const_list, arguments, argument_indices_offset, actual_op_len):
        arguments_ids_list = arguments_ids.tolist()

        predicted_argu = []
        for i, batch_argu_ids in enumerate(arguments_ids_list):
            batch_argu = []
            op_len = actual_op_len[i]
            for i_op, argu_ids_per_op in enumerate(batch_argu_ids):
                if i_op < op_len:
                    op_argu = []
                    for argu_ids in argu_ids_per_op:
                        if argu_ids < len(const_list):
                            cur_argu = const_list[argu_ids]
                            if cur_argu == "none":
                                break
                            op_argu.append(cur_argu)
                        else:
                            op_argu.append(arguments[i][argument_indices_offset[i].index(argu_ids - len(const_list))])
                else:
                    break
                batch_argu.append(op_argu)
            predicted_argu.append(batch_argu)
        
        return predicted_argu
    
    @staticmethod
    def program_ids_to_token(program_ids, op_list, const_list, argument_indices_offset):
        program_ids_list = program_ids

        reserved_token_list = op_list + const_list
        predicted_program_token = []
        for batch_i, batch_prog_ids in enumerate(program_ids_list):
            batch_prog = []
            for ids in batch_prog_ids:
                if ids < len(reserved_token_list):
                    batch_prog.append(reserved_token_list[ids])
                else:
                    ids_in_pasage = ids - len(reserved_token_list)
                    ids_in_number = argument_indices_offset[batch_i].index(ids_in_pasage)
                    batch_prog.append(f"n{ids_in_number}")
                
                if batch_prog[-1] == "EOF":
                    break
            predicted_program_token.append(batch_prog)
        
        return predicted_program_token
    
    @staticmethod
    def argu_ids_to_argu_mathqa(arguments_ids, const_list, argument_indices_offset, actual_op_len):
        arguments_ids_list = arguments_ids.tolist()

        predicted_argu = []
        for i, batch_argu_ids in enumerate(arguments_ids_list):
            batch_argu = []
            op_len = actual_op_len[i]
            for i_op, argu_ids_per_op in enumerate(batch_argu_ids):
                if i_op < op_len:
                    op_argu = []
                    for argu_ids in argu_ids_per_op:
                        if argu_ids < len(const_list):
                            cur_argu = const_list[argu_ids]
                            if cur_argu == "none":
                                break
                            op_argu.append(cur_argu)
                        else:
                            argu_ids_in_numbers = argument_indices_offset[i].index((argu_ids - len(const_list)))
                            op_argu.append(f"n{argu_ids_in_numbers}")
                else:
                    break
                batch_argu.append(op_argu)
            predicted_argu.append(batch_argu)
        
        return predicted_argu
        
    @staticmethod
    def calculate_program(operation, arguments, table):
        def process_row(row):
            row_out = []
            invalid_flag = 0

            for num in row:
                num = num.replace("$", "").strip()
                num = num.split("(")[0].strip()

                num = str_to_num_2(num)

                if num == "n/a":
                    invalid_flag = 1
                    break

                row_out.append(num)

            if invalid_flag:
                return "n/a"

            return row_out

        assert len(operation) == len(arguments)
        
        invalid_flag = 0
        cache_dict = {}
        op_step = 0
        this_res = "n/a"
        try:
            for op, argus in zip(operation, arguments):
                argu_dict = {}
                if op == "add" or op == "subtract" or op == "multiply" or op == "divide" or op == "exp" or op == "greater" \
                    or op == "biggest" or op == "smallest" or op == "secondsmallest":
                    for i, arg in enumerate(argus):
                        if type(arg) in [int, float]:
                            # the arg from the arguments should be number type.
                            argu_dict[i] = arg
                        elif "#" in arg:
                            cache_point = int(arg.replace("#", ""))
                            if cache_point not in cache_dict:
                                invalid_flag = 1
                                break
                            argu_dict[i] = cache_dict[cache_point]
                        else:
                            cur_arg = str_to_num_2(arg)
                            if cur_arg == "n/a":
                                invalid_flag = 1
                                break
                            argu_dict[i] = cur_arg
                    
                    if invalid_flag == 1:
                        break
                    
                    if len(argu_dict) == 0:
                        invalid_flag == 1
                        break
                    
                    if op == "add":
                        this_res = sum([v for v in argu_dict.values()])
                    elif op == "subtract":
                        argu_values = list(argu_dict.values())
                        this_res = argu_values[0]
                        for v in argu_values[1:]:
                            this_res -= v
                    elif op == "multiply":
                        this_res = 1
                        for v in argu_dict.values():
                            this_res *= v
                    elif op == "divide":
                        argu_values = list(argu_dict.values())
                        this_res = argu_values[0]
                        for v in argu_values[1:]:
                            if v == 0:
                                invalid_flag = 1
                                break
                            this_res /= v
                    elif op == "exp":
                        if len(argu_dict) == 2:
                            argu_values = list(argu_dict.values())
                        else:
                            invalid_flag = 1
                            break
                        try:
                            this_res = argu_values[0] ** argu_values[1]
                        except OverflowError:
                            # sometimes the exponential number is too large.
                            invalid_flag = 1
                            break
                    elif op == "greater":
                        if len(argu_dict) == 2:
                            argu_values = list(argu_dict.values())
                            this_res = "yes" if argu_values[0] > argu_values[1] else "no"
                        else:
                            invalid_flag = 1
                            break
                    elif op == "biggest":
                        if len(argu_dict) >= 2:
                            argu_values = list(argu_dict.values())
                            this_res = max(argu_values)
                        else:
                            invalid_flag = 1
                            break
                    elif op == "smallest":
                        if len(argu_dict) >= 2:
                            argu_values = list(argu_dict.values())
                            this_res = min(argu_values)
                        else:
                            invalid_flag = 1
                            break
                    elif op == "secondsmallest":
                        if len(argu_dict) >= 2:
                            argu_values = list(argu_dict.values())
                            smalles_num = min(argu_values)
                            argu_values.pop(smalles_num)
                            this_res = min(argu_values)
                        else:
                            invalid_flag = 1
                            break

                    cache_dict[op_step] = this_res
                    
                elif "table" in op:
                    if table == None:
                        invalid_flag = 1
                        break
                    else:
                        table_dict = {}
                        for row in table:
                            table_dict[row[0]] = row[1:]
                        
                        try:
                            arg = argus[0]
                        except IndexError:
                            invalid_flag = 1
                            break

                        if arg not in table_dict:
                            invalid_flag = 1
                            break
                        
                        cal_row = table_dict[arg]
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
                        
                        cache_dict[op_step] = this_res
            
                op_step +=1

            if this_res != "yes" and this_res != "no" and this_res != "n/a":
                this_res = round(this_res, 5)
        
        except:
            invalid_flag = 1

        return invalid_flag, this_res
    
    @staticmethod
    def compare_program(operation, arguments, golden_op, golden_arguments, id=None):
        golden_op = golden_op[:-1]
        golden_arguments = golden_arguments[:-1]

        invalid_flag = 0

        # 1. the length of the operations shoule be equal
        if len(golden_op) != len(operation):
            invalid_flag = 1
            return invalid_flag
        
        for op_i, op in enumerate(golden_op):
            # 2. the op at each step should be equal
            pred_op = operation[op_i]
            if op != pred_op:
                invalid_flag = 1
                return invalid_flag
            
            gold_op_argu = golden_arguments[op_i]
            pred_op_argu = arguments[op_i]
            gold_op_argu_str = [str_to_num_2(str(argu), convert_const=True) for argu in gold_op_argu]
            pred_op_argu_str = [str_to_num_2(str(argu), convert_const=True) for argu in pred_op_argu]

            if len(gold_op_argu) != len(pred_op_argu):
                if gold_op_argu[-1] == "none":
                    pass
                else:
                    invalid_flag = 1
                    return invalid_flag
            if op in ["add", "multiply", "biggest", "smallest"]:
                for gold_argu in gold_op_argu_str:
                    if gold_argu not in pred_op_argu_str:
                        invalid_flag = 1
                        return invalid_flag
            elif op in ["subtract", "divide", "exp", "greater"]:
                for argu_i, gold_argu in enumerate(gold_op_argu_str):
                    if gold_argu != pred_op_argu_str[argu_i]:
                        invalid_flag = 1
                        return invalid_flag
            elif "table" in op:
                if gold_op_argu[0] != pred_op_argu[0]:
                    invalid_flag = 1
                    return invalid_flag
        
        return invalid_flag
    
    @staticmethod
    def compare_program_mathqa(operation, arguments, golden_op, golden_arguments):
        golden_op = golden_op[:-1]
        golden_arguments = golden_arguments[:-1]

        invalid_flag = 0

        # 1. the length of the operations shoule be equal
        if len(golden_op) != len(operation):
            invalid_flag = 1
            return invalid_flag
        
        for op_i, op in enumerate(golden_op):
            # 2. the op at each step should be equal
            pred_op = operation[op_i]
            if op != pred_op:
                invalid_flag = 1
                return invalid_flag

            gold_op_argu = golden_arguments[op_i]
            pred_op_argu = arguments[op_i]

            if len(gold_op_argu) != len(pred_op_argu):
                invalid_flag = 1
                return invalid_flag

            if op in ["add", "multiply", "biggest", "smallest", "secondsmallest"]:
                for gold_argu in gold_op_argu:
                    if gold_argu not in pred_op_argu:
                        invalid_flag = 1
                        return invalid_flag
            elif op in ["subtract", "divide", "exp", "greater"]:
                for argu_i, gold_argu in enumerate(gold_op_argu):
                    if gold_argu != pred_op_argu[argu_i]:
                        invalid_flag = 1
                        return invalid_flag

        return invalid_flag