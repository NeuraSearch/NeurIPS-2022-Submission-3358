# coding:utf-8

import os
import json
import torch
import codecs
import wandb
import random

from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from utils import set_environment, create_logger
from models.batch_generator import DataBatchGenerator
from models.finqa_model import FinQAModel
from models.FinQANet import Bert_model
from models.model_wrapper import ModelWrapper
from configurations import add_path_relevant_args, \
                            add_model_relevant_args, \
                                 add_train_relevant_args, \
                                     add_assist_args

"""Parse the Arguments"""
parser = ArgumentParser("Train for FinQA.")
add_path_relevant_args(parser)
add_model_relevant_args(parser)
add_train_relevant_args(parser)
add_assist_args(parser)
args = parser.parse_args()
args.t_bsz = args.t_bsz // args.gradient_accumulation_steps

data_name = args.data_name
is_kfold_train = False
if data_name == "finqa":
    from configurations import CONST_LIST as const_list
    from configurations import OPERATION_LIST as op_list
elif data_name == "mathqa":
    from configurations import CONST_LIST_MATHQA as const_list
    from configurations import OPERATION_LIST_MATHQA as op_list
elif data_name =="drop_annotated":
    from configurations import CONST_LIST_DROP_ANNOTATED as const_list
    from configurations import OPERATION_LIST_DROP_ANNOTATED as op_list
    is_kfold_train = True
else:
    raise NotImplementedError(f"Unknown data name: {data_name}")
# sorry, this file is designed for "drop_annotated".
assert data_name == "drop_annotated"

# if True, means using the baseline FinQANet.
is_program_as_sequence = bool(args.is_program_as_sequence)
if is_program_as_sequence:
    from configurations import OPERATION_LIST_MATHQA_2 as op_list
    from configurations import CONST_LIST_MATHQA_2 as const_list

"""Save the Training Configurations"""
conf_save_path = Path(args.model_save_dir) / "configuration.json"
with codecs.open(conf_save_path, "w") as file:
    json.dump(vars(args), file, indent=2)

"""Check GPU Available"""
args.cuda = torch.cuda.device_count() > 0
set_environment(args.seed, args.cuda)

"""Create Logger"""
logger = create_logger(name=f"{data_name}", log_dir=args.model_save_dir)

"""Set Wandb"""
args.wandb = True if int(args.wandb) != 0 else False
if args.wandb:
    raise ValueError("If you want to WANDB, please specify your information below.")
    # wandb.login(key="SPECIFY YOUR KEY HERE.")
    # wandb.init(project=f"{data_name}_{args.plm}", config=args)

"""Main function to execute train"""
def main():
    logger.info("start to train...")

    # create data batch
    train_data_batch_generator = DataBatchGenerator(args, "train", op_list, const_list)
    all_datas = train_data_batch_generator.datas
    random.shuffle(all_datas)

    kfold_datas = []
    for k_idx in range(0, 10):
        this_fold_train = all_datas[0 : k_idx*100] + all_datas[k_idx*100 + 100 : ]
        this_fold_test = all_datas[k_idx*100 : k_idx * 100 + 100]
        assert len(this_fold_train) == 900
        assert len(this_fold_test) == 100
        kfold_datas.append((this_fold_train, this_fold_test))
    
    for k_idx, one_fold_data in enumerate(kfold_datas):
        logger.info(f"START FOLD: {k_idx}.")

        train_data_loader = train_data_batch_generator.data_loader_kfold(
            use_cuda=args.cuda,
            fold_data=one_fold_data[0],
            shuffle_or_not=True)
        
        dev_data_loader = train_data_batch_generator.data_loader_kfold(
            use_cuda=args.cuda,
            fold_data=one_fold_data[1],
            shuffle_or_not=False)       

        args.max_op_len = train_data_batch_generator.max_op_len
        args.max_argu_len = train_data_batch_generator.max_op_argu_len

        # update the train_steps (or backpropagation steps)
        num_train_steps = int(len(train_data_loader) / args.gradient_accumulation_steps) * args.max_epoch
        logger.info(f"Total Number of Training Steps: {num_train_steps}")

        model = FinQAModel(args=args, op_list=op_list, const_list=const_list)
 
        # set interval for wandb to monitor the model
        if args.wandb:
            wandb.watch(model, log_freq=args.log_per_updates)

        # wrap the model
        wrapped_model = ModelWrapper(
            args=args,
            model=model,
            num_train_steps=num_train_steps,
            logger=logger,
            mode="train",
            is_program_as_sequence=is_program_as_sequence,
        )

        if wrapped_model.restore_from_prev:
            epoch_start = wrapped_model.epoch
            assert epoch_start < args.max_epoch + 1
            num_train_steps = (args.max_epoch + 1 - epoch_start) * len(train_data_loader) / args.gradient_accumulation_steps
            best_prog_acc = wrapped_model.last_best_prog_accu
        else:
            epoch_start = 1
            best_prog_acc = float("-inf")
    
        train_start_time = datetime.now()
        last_model_save_step = -1
        for epoch in range(epoch_start, args.max_epoch + 1):
            wrapped_model.avg_reset()
            logger.info(f"FOLD: {k_idx} Epoch: {epoch}")

            for batch in train_data_loader:
                # forward
                if not is_program_as_sequence:
                    wrapped_model.update(batch)
                else:
                    wrapped_model.update_finqa_baseline(batch, op_list, const_list)

                # display loss
                if wrapped_model.updates % args.log_per_updates == 0 or wrapped_model.step == 1:
                    logger.info(
                        "FOLD: [{}] Updates [{}] loss: [{:.2f}] op_loss: [{:.2f}] argu_loss: [{:.2f}] \
                        LR: [{:.6f}] Remaining_Time: [{}]".format(
                            k_idx, wrapped_model.updates, wrapped_model.train_loss.avg, wrapped_model.op_loss.avg, wrapped_model.argu_loss.avg,
                            wrapped_model.lr,
                            str((datetime.now() - train_start_time) / (wrapped_model.updates) * (num_train_steps - wrapped_model.updates)).split(".")[0]
                        )
                    )
                    if args.wandb:
                        wandb.log(
                            {"loss": wrapped_model.train_loss.avg,
                            "op_loss": wrapped_model.op_loss.avg,
                            "argu_loss": wrapped_model.argu_loss.avg,
                            "lr": wrapped_model.lr,}
                        )
                    wrapped_model.avg_reset()
                
                # save model
                if wrapped_model.step == 1 or wrapped_model.step % args.save_every_steps == 0:
                    logger.info(f"FOLD: {k_idx} save model at step {wrapped_model.step}.")
                    save_prefix = os.path.join(args.model_save_dir, f"{k_idx}_checkpoint_normal_{str(wrapped_model.step)}")
                    if last_model_save_step > 0:
                        os.remove(os.path.join(args.model_save_dir, f"{k_idx}_checkpoint_normal_{last_model_save_step}.pt"))
                        os.remove(os.path.join(args.model_save_dir, f"{k_idx}_checkpoint_normal_{last_model_save_step}.ct"))
                        os.remove(os.path.join(args.model_save_dir, f"{k_idx}_checkpoint_normal_{last_model_save_step}.op"))
                        os.remove(os.path.join(args.model_save_dir, f"{k_idx}_checkpoint_normal_{last_model_save_step}.lr"))     
                    last_model_save_step = wrapped_model.step
                    wrapped_model.save(save_prefix, epoch, best_prog_acc)
            
            # after one epoch, start to evaluate
            logger.info(f"FOLD: {k_idx} Epoch: {epoch} start to evaluate...")
            all_predictions = {}
            all_wrong_predictions = {}
            for batch_eval in dev_data_loader:
                if not is_program_as_sequence:
                    predictions, wrong_predictions = wrapped_model.evaluate(batch_eval, op_list, const_list, False)
                else:
                    predictions, wrong_predictions = wrapped_model.evaluate_finqa_baseline(batch_eval, op_list, const_list)
                all_predictions.update(predictions)
                all_wrong_predictions.update(wrong_predictions)
            
            eval_loss = wrapped_model.eval_loss.avg
            exec_accu = wrapped_model.exec_accu.avg
            prog_accu = wrapped_model.prog_accu.avg
            logger.info(f"FOLD: {k_idx} Epoch: {epoch} Eval {len(dev_data_loader) * args.e_bsz} examples Eval_loss: {eval_loss:.2f} Exec_accu: {exec_accu:.2f} Prog_accu: {prog_accu:.2f}")

            if prog_accu > best_prog_acc:
                save_prefix = os.path.join(args.model_save_dir, f"{k_idx}_checkpoint_best_{prog_accu:.2f}")
                wrapped_model.save(save_prefix, epoch, prog_accu)
                best_prog_acc = prog_accu
                logger.info(f"FOLD: {k_idx} Best Prog Accu update: {best_prog_acc:.2f} at epoch: {epoch}.")

            eval_results_save_path = os.path.join(args.eval_results_dir, f"{k_idx}_predictions_{epoch}.json")
            eval_wrong_results_save_path = os.path.join(args.eval_results_dir, f"{k_idx}_predictions_wrong{epoch}.json")
            try:
                with codecs.open(eval_results_save_path, "w") as file:
                    json.dump(all_predictions, file, indent=2)
                with codecs.open(eval_wrong_results_save_path, "w") as file:
                    json.dump(all_wrong_predictions, file, indent=2)
            except TypeError:
                continue

if __name__ == "__main__":
    main()