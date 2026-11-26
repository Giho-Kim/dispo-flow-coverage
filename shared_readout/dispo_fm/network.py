"""Conditional linear U-Net for DiSPO flow matching."""
import math

import torch
from torch import nn


class ConditionalUNet(nn.Module):
    def __init__(self, dimension, context_dim, embed, downs, generator, dtype):
        super().__init__()
        self.downs = tuple(downs)
        self.layers = nn.ModuleDict()
        frequency = torch.arange(embed // 2, dtype=dtype)
        frequency = torch.exp(frequency * (-math.log(10000) / max(embed // 2 - 1, 1)))
        self.register_buffer('frequency', 999 * frequency, persistent=False)

        def dense(name, in_features, out_features):
            # Skip nn.Linear's default initialization since we use Gaussian fan-in.
            layer = nn.Linear(in_features, out_features, device='meta', dtype=dtype)
            layer.to_empty(device='cpu')
            with torch.no_grad():
                layer.weight.normal_(std=in_features ** -.5, generator=generator)
                layer.bias.zero_()
            self.layers[name] = layer

        def block(name, in_features, width):
            dense(f'{name}/dense', in_features, width)
            self.layers[f'{name}/ln'] = nn.LayerNorm(width, eps=1e-6, dtype=dtype)

        def residual(name, in_features, width):
            block(f'{name}/first', in_features, width)
            dense(f'{name}/film', 2 * embed, 2 * width)
            block(f'{name}/second', width, width)
            dense(f'{name}/skip', in_features, width)

        width = 2 * max(embed, context_dim)
        block('condition/0', context_dim, width)
        block('condition/1', width, width)
        dense('condition/2', width, embed)
        incoming = dimension
        for i, width in enumerate(self.downs):
            residual(f'down/{i}/0', incoming, width)
            residual(f'down/{i}/1', width, width)
            incoming = width
        for j in range(2):
            residual(f'mid/{j}', incoming, incoming)
        for i, (width, skip) in enumerate(zip(reversed(self.downs[:-1]), reversed(self.downs[1:]))):
            residual(f'up/{i}/0', incoming + skip, width)
            residual(f'up/{i}/1', width, width)
            incoming = width
        block('out/0', incoming, self.downs[0])
        dense('out/1', self.downs[0], dimension)

    def time_embedding(self, t):
        angles = t.reshape(-1, 1) * self.frequency
        return torch.cat((angles.sin(), angles.cos()), dim=1)

    def context_embedding(self, context):
        context = self.block(context, 'condition/0')
        context = self.block(context, 'condition/1')
        return self.layers['condition/2'](context)

    def block(self, x, name):
        x = self.layers[f'{name}/dense'](x)
        return torch.nn.functional.silu(self.layers[f'{name}/ln'](x))

    def residual(self, x, condition, name):
        y = self.block(x, f'{name}/first')
        gamma, beta = self.layers[f'{name}/film'](condition).chunk(2, dim=1)
        y = self.block(torch.addcmul(beta, gamma, y), f'{name}/second')
        return y + self.layers[f'{name}/skip'](x)

    def velocity(self, x, time_embedding, context_embedding):
        time_embedding = time_embedding.expand(len(context_embedding), -1)
        condition = torch.cat((time_embedding, context_embedding), dim=1)
        skips = []
        for i in range(len(self.downs)):
            for j in range(2):
                x = self.residual(x, condition, f'down/{i}/{j}')
            skips.append(x)
        for j in range(2):
            x = self.residual(x, condition, f'mid/{j}')
        for i in range(len(self.downs) - 1):
            x = torch.cat((x, skips.pop()), dim=1)
            for j in range(2):
                x = self.residual(x, condition, f'up/{i}/{j}')
        return self.layers['out/1'](self.block(x, 'out/0'))

    def forward(self, x, t, context):
        return self.velocity(x, self.time_embedding(t), self.context_embedding(context))

    def checkpoint_parameters(self):
        """Use the v2 names and (input, output) weight layout on disk."""
        for name, layer in self.layers.items():
            if isinstance(layer, nn.Linear):
                yield f'{name}/w', layer.weight, True
                yield f'{name}/b', layer.bias, False
            else:
                yield f'{name}/scale', layer.weight, False
                yield f'{name}/bias', layer.bias, False
