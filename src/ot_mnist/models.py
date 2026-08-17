import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    if t.ndim == 0:
        t = t[None]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device).float() / max(half, 1)
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()
        groups1 = min(8, in_ch)
        groups2 = min(8, out_ch)
        self.norm1 = nn.GroupNorm(groups1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups2, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SmallUNet(nn.Module):
    """Small two-time U-Net for MNIST.

    Input/output are [B, 1, 28, 28]. Conditioning uses t and dt=t-r.
    """

    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        base_ch=32,
        channel_mults=(1, 2, 2),
        time_dim=128,
        dropout=0.0,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        ch = base_ch
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skips = [ch]
        for level, mult in enumerate(channel_mults):
            out = base_ch * mult
            self.down_blocks.append(ResBlock(ch, out, time_dim, dropout))
            ch = out
            self.down_blocks.append(ResBlock(ch, out, time_dim, dropout))
            skips.extend([ch, ch])
            if level != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                skips.append(ch)

        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out = base_ch * mult
            for _ in range(2):
                skip_ch = skips.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out, time_dim, dropout))
                ch = out
            if level != 0:
                self.upsamples.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)

    def make_temb(self, r, t):
        if r.ndim == 0:
            r = r.expand(t.shape[0])
        if t.ndim == 0:
            t = t.expand(r.shape[0])
        dt = t - r
        emb = torch.cat(
            [timestep_embedding(t, self.time_dim), timestep_embedding(dt, self.time_dim)],
            dim=-1,
        )
        return self.time_mlp(emb)

    def forward(self, z, r, t):
        temb = self.make_temb(r, t)
        h = self.in_conv(z)
        hs = [h]
        down_i = 0
        for block_i, block in enumerate(self.down_blocks):
            h = block(h, temb)
            hs.append(h)
            if block_i % 2 == 1 and down_i < len(self.downsamples):
                h = self.downsamples[down_i](h)
                hs.append(h)
                down_i += 1

        h = self.mid2(self.mid1(h, temb), temb)

        up_i = 0
        for block_i, block in enumerate(self.up_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, temb)
            if block_i % 2 == 1 and up_i < len(self.upsamples):
                h = self.upsamples[up_i](h)
                up_i += 1

        return self.out_conv(F.silu(self.out_norm(h)))


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, return_features=False):
        h = self.conv(x)
        h = h.flatten(1)
        feat = F.silu(self.fc1(h))
        logits = self.fc2(feat)
        if return_features:
            return logits, feat
        return logits


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(state[k].detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]

    def copy_to(self, model):
        state = model.state_dict()
        backup = {}
        for k, v in self.shadow.items():
            backup[k] = state[k].detach().clone()
            state[k].copy_(v.to(state[k].device))
        return backup

    def restore(self, model, backup):
        state = model.state_dict()
        for k, v in backup.items():
            state[k].copy_(v.to(state[k].device))

