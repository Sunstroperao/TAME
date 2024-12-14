from torch import nn
import torch.nn.functional as F
import torch


class Expert(nn.Module):
    def __init__(self, input_size, output_size):

        super(Expert, self).__init__()
        self.expert_layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.expert_layer(x)
        return output


class Expert_Gate(nn.Module):

    def __init__(self, input_size, output_size=64, num_experts=4, num_tasks=2):

        super(Expert_Gate, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Define the expert layers
        self.expert_layers = nn.ModuleList([Expert(
            input_size, output_size) for _ in range(num_experts)])

        # Define the gate layer
        self.gate_layers = nn.ModuleList([nn.Sequential(nn.Linear(
            input_size, num_experts), nn.Softmax(dim=2)) for _ in range(num_tasks)])

    def forward(self, x):
        # (bs, seq_len, num_experts, output_size)
        expert_outputs = torch.stack(
            [expert_layer(x) for expert_layer in self.expert_layers], dim=2)
        # (bs, num_tasks, seq_len, num_experts)
        final_outputs = []
        for i, gate_layer in enumerate(self.gate_layers):
            gate_output = gate_layer(x)
            gate_output = gate_output.unsqueeze(-1)

            weighted_expert_outputs = torch.matmul(
                expert_outputs.transpose(2, 3), gate_output)
            weighted_expert_outputs = weighted_expert_outputs.squeeze(-1)
            final_outputs.append(weighted_expert_outputs)
        finally_output = torch.cat(final_outputs, dim=-1)
        return finally_output


if __name__ == '__main__':
    expert = Expert_Gate(6, 32)
    x = torch.randn(256, 16, 6)

    y = expert(x)
    y = nn.GELU()(y)
    print(y.shape)
