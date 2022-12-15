# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

class Attention(nn.Module):
    def __init__(self, h_dim):
        super(Attention, self).__init__()

        self.q_proj = nn.Linear(h_dim, h_dim, bias=False)
        self.k_proj = nn.Linear(h_dim, h_dim, bias=False)
        self.v_proj = nn.Linear(h_dim, h_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        score = torch.matmul(
            q,
            torch.permute(k, (0, 2, 1))
        )
            
        if mask is not None:
            mask = torch.unsqueeze(mask, 1)
            score = score.masked_fill_((1 - mask).bool(), -99999.)

        attn_weights = F.softmax(score, dim=-1)

        # (b, 1, s) * (b, s, h) -> (b, 1, h)
        context_vector = torch.bmm(attn_weights, v)
        
        return context_vector

decoder_output = namedtuple("output", "operations_logits, arguments_logits, operations_ids, arguments_ids")

class HierarchicalDecoder(nn.Module):
    def __init__(self,
                 op_len,
                 const_len,
                 h_size,
                 n_layers,
                 op_list,
                 const_list,
                 dropout_p,
                 max_op_len,
                 max_argu_len,
                 eos_id,
                 help_op_with_one_argu):
        super(HierarchicalDecoder, self).__init__()

        self.op_list = op_list
        self.const_list = const_list
        self.max_op_len = max_op_len
        self.max_argu_len = max_argu_len
        self.eos_id = eos_id
        self.help_op_with_one_argu = help_op_with_one_argu

        # create the operations and numbers sequences
        self.op_sequence = nn.Parameter(
            torch.arange(0, op_len),
            requires_grad=False,
        )

        self.const_sequence = nn.Parameter(
            torch.arange(0, const_len),
            requires_grad=False,
        )
        
        # create the embedding for the <go>
        self.go_embedding = nn.Parameter(
            torch.randn(1, h_size),
            requires_grad=True,
        )

        # create embeddings
        self.op_embeddings = nn.Embedding(
            op_len, h_size
        )
        self.const_embeddings = nn.Embedding(
            const_len, h_size
        )

        # create RNN
        self.n_layers = n_layers
        self.rnn_op = nn.GRU(
            input_size=h_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p
        )
        self.rnn_argu = nn.GRU(
            input_size=h_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p
        )

        # create linear combination
        self.op_c_combine = nn.Linear(
            2 * h_size, h_size, bias=True,
        )
        self.op_c_combine_layer_norm = nn.LayerNorm([1, h_size])
        self.num_c_combine = nn.Linear(
            2 * h_size, h_size, bias=True,
        )
        self.num_c_combine_layer_norm = nn.LayerNorm([1, h_size])

        # create Attention
        self.op_attention = Attention(h_size)
        self.num_attention = Attention(h_size)
        self.update_attention = Attention(h_size)

        # create output projection
        self.op_pred_output = nn.Linear(h_size, h_size, bias=True)
        self.num_pred_output = nn.Linear(h_size, h_size, bias=True)

        # create h_op -> h_num linear
        self.op_to_num = nn.Linear(h_size, h_size)
        self.pred_op_emb_to_num = nn.Linear(h_size, h_size)
        self.op_and_emb_comb = nn.Linear(2 * h_size, h_size)
        self.op_and_emb_comb_layer_norm = nn.LayerNorm([1, h_size])

    def forward(self,
                input_mask,
                input_question_mask,
                input_number_mask,
                encoder_outputs,
                golden_op=None,
                golden_argu=None,
                is_train=True,
                sampling_rate=1.0):

        bsz, _, h_size = encoder_outputs.size()
        max_op_len = golden_op.size(1) if is_train else self.max_op_len
        max_argu_len = golden_argu.size(2) if is_train else self.max_argu_len

        # Tuple contains the g_op in each time step (b, 1)
        if is_train:
            split_op_ids = torch.split(golden_op, 1, dim=1)
            # split_argu_ids = torch.split(golden_argu, 1, dim=1)
            # split_argu_ids = torch.split(golden_argu, self.max_argu_len, dim=1)
            # [bsz, 1, max_argu_len] * max_op_len
            split_argu_ids = torch.split(golden_argu, 1, dim=1)
            assert len(split_op_ids) == len(split_argu_ids)
        else:
            assert sampling_rate == 1.0

        # initialize the embeddings (b, _, h)
        op_embed = self.op_embeddings(self.op_sequence).repeat(bsz, 1, 1)
        const_embed = self.const_embeddings(self.const_sequence).repeat(bsz, 1, 1)
        const_full_mask = torch.ones((bsz, const_embed.size(1))).to(self.op_sequence.device)
        # argument_embed: (b, N_n + S, h)
        argument_embed = torch.cat((const_embed, encoder_outputs), dim=1)
        # (b, N_n + S)
        number_mask = torch.cat((const_full_mask, input_number_mask), dim=1)

        # initialize the previous prediction as <go>
        prev_op_embed = torch.unsqueeze(self.go_embedding, dim=0)
        # [b, 1, h]
        prev_op_embed = prev_op_embed.repeat(bsz, 1, 1)

        # initialize the hidden states for RNN
        prev_op_rnn_h = torch.zeros(
            self.n_layers, bsz, h_size,
            device=encoder_outputs.device,
        )
        prev_argu_rnn_h = torch.zeros(
            self.n_layers, bsz, h_size,
            device=encoder_outputs.device,
        )

        operations_logits = []
        arguments_logits = []
        
        operations_ids = []
        arguments_ids = []

        for op_step in range(max_op_len):

            # attention to encoder_outputs
            # (b, 1, h)
            enc_context = self.op_attention(
                query=prev_op_embed,
                key=encoder_outputs,
                value=encoder_outputs,
                mask=input_question_mask,
            )

            # (b, 1, h)
            op_c_comb = self.op_c_combine(
                torch.cat(
                    (prev_op_embed, enc_context), 
                    dim=2,
                )
            )
            op_c_comb = self.op_c_combine_layer_norm(op_c_comb)
            op_c_comb = F.relu(op_c_comb)

            # o_op_1: (b, 1, h)
            # prev_op_rnn_h: (n_layers, b, h)
            o_op_1, prev_op_rnn_h = self.rnn_op(
                op_c_comb, prev_op_rnn_h,
            )

            # op_embed: (b, N_o, h)
            # o_op_1: (b, 1, h)
            # op_logits: (b, N_0)
            o_op_output = self.op_pred_output(o_op_1)
            o_op_output = F.relu(o_op_output)
            o_op_output = torch.permute(o_op_output, (0, 2, 1))
            op_logtis = torch.matmul(
                op_embed,
                o_op_output,
            )
            op_logtis = torch.squeeze(op_logtis, dim=2)
            operations_logits.append(op_logtis)

            # # FIXME: this is for SVAMP only
            # op_mask = torch.ones((bsz, op_logtis.size(1))).to(op_logtis.device)
            # op_mask[:, 4:] = 0
            # op_mask[:, -1] = 1
            # op_logtis = op_logtis.masked_fill_((1-op_mask).bool(), -99999.)

            if is_train:
                # (b, 1, 1)
                pred_op_ids = torch.unsqueeze(
                    split_op_ids[op_step], dim=1
                )

                # scheduled sampling here
                pred_op_ids_sample = torch.argmax(
                    op_logtis, axis=-1, keepdim=True
                )
                pred_op_ids_sample = torch.unsqueeze(
                    pred_op_ids_sample, dim=-1
                )
                # When train on GPU, the np.random.choice() requires the numpy data,
                # when convert tensor(GPU) to numpy data, it should be converted to the cpu first,
                # pred_op_ids_sample = pred_op_ids_sample.cpu().numpy()
                # this is why use "sample_or_not" here.
                sample_or_not = np.random.choice([0, 1], p=[1.0 - sampling_rate, sampling_rate])
                if sample_or_not == 1:
                    pred_op_ids = pred_op_ids_sample
            else:
                # [b, 1]
                pred_op_ids = torch.argmax(
                    op_logtis, axis=-1, keepdim=True
                )
                # [b, 1, 1]
                pred_op_ids = torch.unsqueeze(
                    pred_op_ids, dim=-1
                )
                operations_ids.append(pred_op_ids)
            
            # [b, 1, h]
            pred_op_ids_expanded = torch.repeat_interleave(
                pred_op_ids, h_size, dim=2
            )
            # [b, 1, h]
            prev_op_embed = torch.gather(
                op_embed, dim=1, index=pred_op_ids_expanded
            )


            num_init_out_1 = self.pred_op_emb_to_num(prev_op_embed)
            num_init_out_1 = F.relu(num_init_out_1)
            num_init_out_2 = self.op_to_num(o_op_1)
            num_init_out_2 = F.relu(num_init_out_2)
            input_argu_embed = self.op_and_emb_comb(
                torch.cat((num_init_out_1, num_init_out_2), dim=-1)
            )
            # [b, 1, h]
            input_argu_embed = self.op_and_emb_comb_layer_norm(input_argu_embed)

            argu_embed_cache = []    # for update #N use
            sub_arguments_logits = []
            sub_arguments_ids = []
            for num_step in range(max_argu_len):
                # we combine input_argu_embed and prev_op_embed information 
                # like we do at the initial step when generating the first argu.

                enc_num_context = self.num_attention(
                    query=input_argu_embed,
                    key=encoder_outputs,
                    value=encoder_outputs,
                    mask=input_mask,
                )

                # (b, 1, h)
                num_c_comb = self.num_c_combine(
                    torch.cat((input_argu_embed, enc_num_context), dim=-1)
                )
                num_c_comb = self.num_c_combine_layer_norm(num_c_comb)
                num_c_comb = F.relu(num_c_comb)
                
                # out_num: (b, 1, h)
                # prev_argu_rnn_h: (n_layers, b, h)
                out_num, prev_argu_rnn_h = self.rnn_argu(
                    num_c_comb, prev_argu_rnn_h,
                )

                # argument_embed: (b, N_n + S, h)
                # out_num: (b, 1, h)
                # op_logits: (b, N_n + S)
                argu_logits = torch.matmul(
                    argument_embed,
                    F.relu(self.num_pred_output(out_num)).permute(0, 2, 1),
                )
                argu_logits = torch.squeeze(argu_logits, dim=2)

                # argu_logits = argu_logits.masked_fill_((1 - number_mask).bool(), -99999.)

                # There's no chance for the #N to be produced, 
                # where N is greater or equal than the op_step.
                after_cache_start = self.const_list.index("#" + str(op_step))
                after_cache_end = self.const_list.index("#" + str(10))
                this_step_mask = number_mask.clone()
                this_step_mask[:, after_cache_start : after_cache_end + 1] = 0.0

                # HACK: some operation only supports one argument,
                #       force these operations generate "none" at 2nd step.
                if self.help_op_with_one_argu and num_step == 1:
                    one_argu_op_temp = (pred_op_ids >= self.op_list.index("floor")).float()
                    one_argu_op_not_eos = (pred_op_ids != self.eos_id).float()
                    # [b, 1, 1] -> [b, 1]
                    one_argu_op_mask = (one_argu_op_temp * one_argu_op_not_eos).squeeze(1)

                    # [b, C+S]
                    this_step_mask_2 = torch.zeros_like(number_mask).to(number_mask.device)
                    this_step_mask_2[:, self.const_list.index("none")] = 1.0

                    this_step_mask = torch.where(one_argu_op_mask == 1, this_step_mask_2, this_step_mask)

                argu_logits = argu_logits.masked_fill_((1 - this_step_mask).bool(), -99999.)

                sub_arguments_logits.append(argu_logits)

                if is_train:
                    # HACK: hard code here.
                    # FIXED: if we support operation with more than 2 steps,
                    #        maybe we could change the split function for the golden num.
                    # (b, 1, 1)
                    # pred_argu_ids = torch.unsqueeze(
                    #     split_argu_ids[op_step * 2 + num_step], dim=1
                    # )
                    pred_argu_ids = torch.unsqueeze(
                        split_argu_ids[op_step][:, :, num_step], dim=1
                    )
                    # scheduled sampling
                    pred_argu_ids_sample = torch.argmax(
                        argu_logits, axis=-1, keepdim=True
                    )
                    # [b, 1, 1]
                    pred_argu_ids_sample = torch.unsqueeze(
                        pred_argu_ids_sample, dim=-1
                    )
                    sample_or_not = np.random.choice([0, 1], p=[1.0 - sampling_rate, sampling_rate])
                    if sample_or_not == 1:
                        pred_argu_ids = pred_argu_ids_sample
                else:
                    # [b, 1]
                    pred_argu_ids = torch.argmax(
                        argu_logits, axis=-1, keepdim=True
                    )
                    # [b, 1, 1]
                    pred_argu_ids = torch.unsqueeze(
                        pred_argu_ids, dim=-1
                    )
                    sub_arguments_ids.append(pred_argu_ids)

                # [b, 1, h]
                pred_argu_ids_expand = torch.repeat_interleave(
                    pred_argu_ids, h_size, dim=2
                )
                # [b, 1, h]
                input_argu_embed = torch.gather(
                    argument_embed, dim=1, index=pred_argu_ids_expand
                )
                argu_embed_cache.append(torch.squeeze(input_argu_embed, dim=1))
            
            sub_arguments_logits = torch.stack(sub_arguments_logits, dim=1)
            arguments_logits.append(sub_arguments_logits)

            if not is_train:
                # [bsz, 1, max_argu_len]
                sub_arguments_ids = torch.stack([sub_argu.squeeze(1) for sub_argu in sub_arguments_ids], dim=-1)
                arguments_ids.append(sub_arguments_ids)

            # # update the #N
            # # (b, max_argu_len, h)
            # this_op_const_embed = torch.stack(argu_embed_cache, dim=1)
            # # (b, 1, h)
            # update_cache = self.update_attention(
            #     query=prev_op_embed,
            #     key=this_op_const_embed,
            #     value=this_op_const_embed,             
            # )

            # # (b, N_n + S, h)
            # update_cache = update_cache.repeat(1, number_mask.size(1), 1)
            # (1, N_n + S)
            this_update_mask = torch.zeros((1, number_mask.size(1))).to(number_mask.device)
            update_index = (self.const_list.index("#" + str(op_step)))
            this_update_mask[:, update_index] = 1.0
            # (1, N_n + S, 1) -> (b, N_n + S, h)
            this_update_mask = torch.unsqueeze(
                this_update_mask, dim=2
            ).repeat(bsz, 1, h_size)

            # use naive update cache
            update_cache = out_num.repeat(1, number_mask.size(1), 1)
            
            # (b, N_n + S, h)
            argument_embed = torch.where(
                this_update_mask > 0, update_cache, argument_embed
            )
        
        operations_logits = torch.stack(operations_logits, dim=1)
        arguments_logits = torch.stack(arguments_logits, dim=1)
        
        if len(operations_ids) != 0 and len(arguments_ids) != 0:
            operations_ids = torch.stack([op.squeeze(1) for op in operations_ids], dim=1).squeeze(-1)
            arguments_ids = torch.stack([sub_argu.squeeze(1) for sub_argu in arguments_ids], dim=1).squeeze(-1)

        outputs = decoder_output(
            operations_logits=operations_logits,
            arguments_logits=arguments_logits,
            operations_ids=operations_ids,
            arguments_ids=arguments_ids,
        )

        return outputs

if __name__ == "__main__":
    const_list = ["const_1", "const_2", "const_3", "const_4", "const_5", "const_6", "const_7", "const_8", "const_9", "const_10",
                "const_100", "const_1000", "const_10000", "const_100000", "const_1000000", "const_10000000",
                "const_1000000000", "const_M1", "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10", 
                "none"]

    op_list = ["add", "subtract", "multiply", "divide", "exp", "greater", 
                    "table_sum", "table_average", "table_max", "table_min",
                    "EOF"]

    decoder = HierarchicalDecoder(
        op_len=5,
        const_len=10,
        h_size=12,
        n_layers=1,
        const_list=["1", "2", "3", "4", "5", "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10"],
        dropout_p=0.1,
        max_op_len=5,
        max_argu_len=4,
        eos_id=op_list.index("EOF")
    )
    
    dummy_encoder_outputs = torch.randn((2, 10, 12))
    dummy_input_mask = torch.FloatTensor(
                        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
                        )
    dummy_input_number_mask = torch.FloatTensor(
                        [[0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 1, 0, 0, 0, 0]]
                         )
    dummy_golden_op = torch.tensor(
                        [[1, 2, 3],
                        [4, 0, 0]], dtype=torch.long,
                        )
    dummy_golden_num = torch.tensor(
                        [[[3, 5, 0, 0], [12, 2, 3, 5], [8, 18, 2, 0]],
                        [[15, 16, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.long,
                        )

    # train
    outputs = decoder(
        input_mask=dummy_input_mask,
        input_number_mask=dummy_input_number_mask,
        encoder_outputs=dummy_encoder_outputs,
        golden_op=dummy_golden_op,
        golden_argu=dummy_golden_num,
        is_train=True,
    )

    print(outputs.operations_logits.size())
    print(outputs.arguments_logits.size())

    # test
    outputs = decoder(
        input_mask=dummy_input_mask,
        input_number_mask=dummy_input_number_mask,
        encoder_outputs=dummy_encoder_outputs,
        is_train=False,
    )

    print(outputs.operations_logits.size())
    print(outputs.arguments_logits.size())