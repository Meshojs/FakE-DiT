"""
Training a DiT aka Diffusion Transformer model on MNIST
Following my model FakE but using better Arch -> FakE-DiT
"""

import math

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class SpaceTimeSteps(nn.Module):
    """
    Disc :
        Given the number of T/TotalSteps
        Instead of going through step 1,2,3 etc
        we choose N steps
    Example :
         N = 4 = number of steps
    Return :
        t : number of steps

    """
    def __init__(self, T, N):
        super().__init__()
        self.T: int = T
        self.N: int = N

    def forward(self) -> list[int]:
        assert self.T != 0
        assert self.N != 0
        jumps = (self.T - 1) / (self.N - 1)
        _c = 0.0
        t: list[int] = []
        for _ in range(self.N):
            t.append(round(_c))
            _c = _c + jumps
        return t

class TimeEmbedding(nn.Module):
    """
    Disc :
        We are looking to convert every step t
        to a vector representation
    Example :
        Step 100 -> [sin , cos , sin , ...]
    Return :
        Position of all timesteps t
    """

    def __init__(self, T, dim=64):
            super().__init__()
            self.dim = dim
            self.T = T
            half = dim // 2
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
            self.register_buffer("freqs", freqs, persistent=False)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
    def forward(self, t):
       args = t.float()[:, None] * self.freqs[None]
       emb = torch.cat([args.sin(), args.cos()], dim=-1)
       return self.mlp(emb)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, image_size, patch_size):
        super().__init__()
        N = (image_size // patch_size) ** 2
        pe = torch.zeros(N, d_model)                  # one row per patch

        for emb in range(N):
            for i in range(d_model // 2):
                angle = emb / (10000 ** ((2 * i) / d_model))
                pe[emb, 2 * i] = math.sin(angle)
                pe[emb, 2 * i + 1] = math.cos(angle)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):                             # x: (B, N, d_model)
        return x + self.pe

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, patch_size, d_model):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                      # (B, 3, H, W)
        x = self.proj(x)                       # (B, d_model, H/p, W/p)
        return x.flatten(2).transpose(1, 2)    # (B, N, d_model)

class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            batch_first=True
        )

        self.norm2 = nn.RMSNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.adaLN = nn.Linear(d_model, 6 * d_model)

        # Critical DiT initialization
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, c):

        shift1, scale1, gate1, \
        shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)

        # Attention
        h = self.norm1(x)

        h = h * (1 + scale1[:, None])
        h = h + shift1[:, None]

        attn_out = self.attn(
            h, h, h,
            need_weights=False
        )[0]

        x = x + gate1[:, None] * attn_out

        # MLP
        h = self.norm2(x)

        h = h * (1 + scale2[:, None])
        h = h + shift2[:, None]

        mlp_out = self.mlp(h)

        x = x + gate2[:, None] * mlp_out

        return x



class FakEDiT(nn.Module):
    def __init__(self, T, N, in_ch, d_model, n_heads, n_blocks, dropout, image_size, patch_size):
        super().__init__()
        self.in_ch = in_ch
        self.patch_size = patch_size
        self.grid = image_size // patch_size

        self.patch_emb = PatchEmbed(in_ch, patch_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, image_size, patch_size)
        self.t_emb = TimeEmbedding(T, d_model)
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads) for _ in range(n_blocks)])

        self.final_norm = nn.RMSNorm(d_model)
        self.final_linear = nn.Linear(d_model, patch_size * patch_size * in_ch)

    def forward(self, x , t):
        x = self.pos_emb(self.patch_emb(x))
        c = self.t_emb(t).to(device)
        for block in self.blocks:
            x = block(x, c).to(device)
        x = self.final_linear(self.final_norm(x))
        return self.unpatchify(x)

    def unpatchify(self, x):
        B = x.shape[0]
        p, g, C = self.patch_size, self.grid, self.in_ch
        x = x.reshape(B, g, g, p, p, C)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, C, g * p, g * p)
