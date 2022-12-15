# coding:utf-8

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import copy
import torch
import torch.nn as nn

from collections import namedtuple
from .hierarchical_decoder import HierarchicalDecoder
from .gnn import GNN

FinQA_Output = namedtuple("FinQAOutput", 
                          "operation_logits, arguments_logits, \
                              operation_ids, arguments_ids, \
                                  op_loss, argu_loss")

class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)

class FinQAModel(nn.Module):
    def __init__(self, args, op_list, const_list):
        super(FinQAModel, self).__init__()

        self.args = copy.deepcopy(args)

        if self.args.plm.startswith("roberta"):
            from transformers import RobertaModel
            from transformers import RobertaConfig
            self.config = RobertaConfig.from_pretrained(self.args.plm)
            self.plm_model = RobertaModel.from_pretrained(self.args.plm)
        elif self.args.plm.startswith("t5"):
            from transformers import T5EncoderModel
            from transformers import T5Config
            self.config = T5Config.from_pretrained(self.args.plm)
            self.plm_model = T5EncoderModel.from_pretrained(self.args.plm)        
        else:
            raise NotImplementedError
        
        # self.gnn_unit = GNN(
        #     h_size=self.config.hidden_size,
        #     n_head=self.args.n_head,
        # )

        # self.gnn_num = GNN(
        #     h_size=self.config.hidden_size,
        #     n_head=self.args.n_head,
        # )

        # self.proj_de_inp = nn.LayerNorm(self.config.hidden_size)

        # self.residual_gru = ResidualGRU(
        #     hidden_size=self.config.hidden_size,
        #     dropout=self.args.dropout_p,
        #     num_layers=2,
        # )

        self.decoder = HierarchicalDecoder(
            op_len=len(op_list),
            const_len=len(const_list),
            h_size=self.config.hidden_size,
            n_layers=self.args.n_layers,
            op_list=op_list,
            const_list=const_list,
            dropout_p=self.args.dropout_p,
            max_op_len=self.args.max_op_len,
            max_argu_len=self.args.max_argu_len,
            eos_id=op_list.index("EOF"),
            help_op_with_one_argu=True if self.args.data_name == "mathqa" else False,
        )

        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self,
                is_training,
                input_ids,
                input_mask,
                segment_ids,
                input_number_unit_graph,
                input_number_number_graph,
                input_number_indices,
                graph_number_mask,
                input_unit_ids,
                input_question_mask,
                input_number_mask,
                golden_op=None,
                golden_op_mask=None,
                golden_arguments=None,
                golden_argu_mask=None,
                requires_loss=True,
                sampling_rate=1.0):
        """
            Args:
                is_training: bool.
                input_ids: [bsz, max_input_len].
                input_mask: [bsz, max_input_len].
                segment_ids: [bsz, max_input_len].
                input_number_unit_graph: [bsz, max_num, 6].
                input_number_number_graph: [bsz, max_num, max_input_len].
                input_number_indices: [bsz, max_num].
                graph_number_mask: [bsz, max_num].
                input_unit_ids: [1, 6].
                input_question_mask: [bsz, max_input_len].
                input_number_mask: [bsz, max_input_len].
                golden_op: [bsz, max_op_len].
                golden_op_mask: [bsz, max_op_len].
                golden_arguments: [bsz, max_op_len, max_op_argu_len].
                golden_argu_mask: [bsz, max_op_len, max_op_argu_len].
                sampling_rate: the probability of use predicted token as next step input token. By dafault, 1.0.
        """
        # bsz = input_ids.size(0)

        # [bsz, max_input_len, h]
        if self.args.plm.startswith("roberta"):
            encoder_outputs = self.plm_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
            )
            # unit_embeddings = self.plm_model.embeddings(input_unit_ids).repeat(bsz, 1, 1)
        elif self.args.plm.startswith("t5"):
            encoder_outputs = self.plm_model(
                input_ids=input_ids,
                attention_mask=input_mask,
            )         
            # unit_embeddings = self.plm_model.shared(input_unit_ids).repeat(bsz, 1, 1)
        else:
            raise ValueError(f"Unknown PLM: {self.args.plm}")

        # _, max_input_len, h_size = encoder_outputs.last_hidden_state.size()

        # """Apply GNN"""
        # # [bsz, 6, h]
        # # [bsz, max_num, h]
        # unit_graph_info = self.gnn_unit(
        #     encoder_outputs=unit_embeddings,
        #     num_graph=input_number_unit_graph,
        # )

        # # [bsz, max_num, h]
        # number_graph_info = self.gnn_num(
        #     encoder_outputs=encoder_outputs.last_hidden_state,
        #     num_graph=input_number_number_graph,
        # )

        # # [bsz, max_num, h]
        # graph_info = unit_graph_info + number_graph_info

        # # [bsz, max_input_len+1, h]
        # just_number_info = torch.zeros(
        #     (bsz, max_input_len+1, h_size),
        #     dtype=torch.float,
        #     device=encoder_outputs.last_hidden_state.device,
        # )

        # # [bsz, max_num]
        # clamped_input_number_indices = input_number_indices.masked_fill_(
        #     (1 - graph_number_mask).bool(),
        #     just_number_info.size(1) - 1,
        # )
        # # [bsz, max_num, h]
        # clamped_input_number_indices_expanded = torch.unsqueeze(clamped_input_number_indices, 2).repeat(1, 1, h_size)

        # just_number_info.scatter_(1, clamped_input_number_indices_expanded, graph_info)
        # just_number_info = just_number_info[:, :-1, :]

        # decoder_input = self.residual_gru(
        #     self.proj_de_inp(
        #         encoder_outputs.last_hidden_state + just_number_info
        #         )
        #     )
        # """Finish GNN"""
        
        decoder_input = encoder_outputs.last_hidden_state

        outputs = self.decoder(
            input_mask=input_mask,
            input_question_mask=input_question_mask,
            input_number_mask=input_number_mask,
            encoder_outputs=decoder_input,
            golden_op=golden_op,
            golden_argu=golden_arguments,
            is_train=is_training,
            sampling_rate=sampling_rate,
        )

        # [bsz, max_op_in_batch, C_op]
        operation_logits = outputs.operations_logits
        # [bsz, max_op_in_batch, max_argu_in_batch, C_argu]
        arguments_logits = outputs.arguments_logits
        operation_ids = outputs.operations_ids
        arguments_ids = outputs.arguments_ids
        
        if requires_loss:
            assert golden_op is not None
            assert golden_op_mask is not None
            assert golden_arguments is not None
            assert golden_argu_mask is not None

            bsz, max_op_len, _ = operation_logits.size()
            # [bsz * max_op_len, ]
            golden_op_flatten = torch.reshape(golden_op, (-1, ))
            # [bsz * max_op_len, C_op]
            operation_logits_reshape = torch.reshape(operation_logits, (bsz * max_op_len, -1))
            # [bsz * max_op_len, ]
            golden_op_mask_flatten = torch.reshape(golden_op_mask, (-1, ))

            op_loss = self.ce_loss(input=operation_logits_reshape, target=golden_op_flatten)
            op_loss = torch.sum(op_loss * golden_op_mask_flatten) / torch.sum(golden_op_mask_flatten)
            
            _, _, max_op_argu_len, _ = arguments_logits.size()
            # [bsz, max_op_len, max_op_argu_len]
            # -> [bsz, max_op_len * max_op_argu_len]
            # -> [bsz * max_op_len * max_op_argu_len, ]
            golden_arguments_flatten = torch.reshape(
                torch.reshape(golden_arguments, (bsz, -1)),
                (-1, )
            )
            # [bsz, max_op_len, max_op_argu_len, C_argu]
            # -> [bsz * max_op_len * max_op_argu_len, C_argu]
            arguments_logits_reshape = torch.reshape(
                torch.reshape(
                    arguments_logits, (bsz, max_op_len * max_op_argu_len, -1)
                ),
                (bsz * max_op_len * max_op_argu_len, -1)
            )
            # [bsz, max_op_len, max_op_argu_len]
            # -> [bsz * max_op_len * max_op_argu_len, ]
            golden_argu_mask_flatten = torch.reshape(
                torch.reshape(
                    golden_argu_mask, (bsz, -1)
                ),
                (-1, )
            )
            argu_loss = self.ce_loss(input=arguments_logits_reshape, target=golden_arguments_flatten)
            argu_loss = torch.sum(argu_loss * golden_argu_mask_flatten) / torch.sum(golden_argu_mask_flatten)
        
        else:
            # dummy losses
            op_loss = torch.zeros(1).to(operation_logits.device)
            argu_loss = torch.zeros(1).to(arguments_logits.device)
        
        return FinQA_Output(
            operation_logits=operation_logits,
            arguments_logits=arguments_logits,
            operation_ids=operation_ids,
            arguments_ids=arguments_ids,
            op_loss=op_loss,
            argu_loss=argu_loss,
        )