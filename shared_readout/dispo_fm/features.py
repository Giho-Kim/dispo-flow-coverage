"""Fixed Fourier state features for continuous-state DiSPO tasks."""

import numpy as np
import torch


FOURIER_DIM = 128


@torch.no_grad()
def fourier_features(observations, device):
    """DiSPO's fixed grouped RandMLP followed by sine, cosine and bias."""
    groups, width = FOURIER_DIM // 2, 32
    rng = torch.Generator().manual_seed(0)

    def kernel(shape, fan_in):
        weight = torch.empty(shape, dtype=torch.float32)
        # Flax's default LeCun initialization uses a truncated normal.
        torch.nn.init.trunc_normal_(weight, a=-2., b=2., generator=rng)
        return (weight / (fan_in ** .5 * .8796256610342398)).to(device)

    first = kernel((observations.shape[1], groups * width), observations.shape[1])
    second = kernel((groups, width, width), width)
    last = kernel((groups, width, 1), width)
    features = np.ones((len(observations), FOURIER_DIM + 1), dtype=np.float32)
    for start in range(0, len(observations), 4096):
        batch = torch.as_tensor(observations[start:start + 4096], device=device)
        hidden = (batch @ first).relu().reshape(-1, groups, width).transpose(0, 1)
        phase = ((hidden @ second).relu() @ last).squeeze(-1).T
        features[start:start + len(batch), :-1] = torch.cat((phase.sin(), phase.cos()), dim=1).cpu().numpy()
    return features
