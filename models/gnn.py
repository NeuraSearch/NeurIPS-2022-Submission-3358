# coding:utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)

class GNN(nn.Module):
    def __init__(self, h_size, n_head):
        super(GNN, self).__init__()

        assert h_size % n_head == 0
        self.n_head = n_head

        self.linear_1 = FFNLayer(h_size, h_size, h_size, 0.0)
        # self.linear_2 = FFNLayer(h_size, h_size, h_size, 0.0)

        # self.conv_linear = {
        #     0: self.linear_1,
        #     1: self.linear_2,
        # }

        self.output_linear = FFNLayer(h_size, h_size, h_size, 0.0)

    @staticmethod
    def gconv(input_x, graph, linear_transform, n_head):
        """
            input_x: [bsz, N, h] or [bsz, 6, h]
            graph: [bsz, N, N] or [bsz, max_num, 6]
            linear_transform: [h, n_head * h_head].
            n_head: int.
        """
        bsz, N, _ = input_x.size()
        max_num = graph.size(1)

        # [bsz, N, h] -> [bsz, N, n_head * h_head]
        input_x = linear_transform(input_x)

        # [bsz, N, n_head, h_head]
        input_x_reshaped = torch.reshape(input_x, (bsz, N, n_head, -1))
        # [bsz, n_head, N, h_head]
        input_x_reshaped = torch.permute(input_x_reshaped, (0, 2, 1, 3))
        # [bsz * n_head, N, h_head]
        input_x_reshaped = torch.reshape(input_x_reshaped, (bsz * n_head, N, -1))

        # [bsz, N, N] -> [bsz, 1, max_num, N]
        graph_input_xpand = torch.unsqueeze(graph, 1)
        # [bsz, n_head, max_num, N]
        graph_input_xpand = graph_input_xpand.repeat(1, n_head, 1, 1)
        # [bsz * n_head, max_num, N]
        graph_input_xpand_reshaped = torch.reshape(graph_input_xpand, (bsz * n_head, max_num, N))

        # [bsz * n_head, max_num, h_head]
        head_graph_info = torch.bmm(graph_input_xpand_reshaped, input_x_reshaped)
        # [bsz, n_head, max_num, h_head]
        head_graph_info = torch.reshape(head_graph_info, (bsz, n_head, max_num, -1))
        # [bsz, n_head, max_num, h_head] -> [bsz, max_num, n_head, h_head] -> [bsz, N, h]
        graph_info = torch.reshape(
            torch.permute(
                head_graph_info, (0, 2, 1, 3)
            ),
            (bsz, max_num, -1)
        )

        return graph_info

    def forward(self,
                encoder_outputs,
                num_graph):
        
        graph_info = encoder_outputs

        # for i in range(2):
        #     print("Graph Info: ", graph_info.size())
        #     graph_info = self.gconv(
        #         input_x=graph_info,
        #         graph=num_graph,
        #         linear_transform=self.conv_linear[i],
        #         n_head=self.n_head
        #     )

        graph_info = self.gconv(
            input_x=graph_info,
            graph=num_graph,
            linear_transform=self.linear_1,
            n_head=self.n_head,
        )

        graph_info_final = self.output_linear(graph_info)
        
        return graph_info_final

if __name__ == "__main__":
    gnn = GNN(30, 6)

    encoder_outputs = torch.FloatTensor(2, 3, 30).fill_(1)
    num_graph = torch.FloatTensor([
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 1, 1]
        ],
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ]
    ])

    out = gnn(encoder_outputs, num_graph)
    print(out.size())

    a = torch.ones((2, 3, 5))
    b = torch.zeros((2, 3, 5))
    c = torch.tensor(
        [[1, 1, 0], [0, 1, 0]]
    )
    c_ = torch.unsqueeze(c, 2).repeat(1, 1, 5)

    d = torch.where(c_ == 1, a, b)
    print(d)