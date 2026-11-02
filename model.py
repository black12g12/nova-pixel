import torch
from torch import nn
from config import TEXT_VOCAB_SIZE, TEXT_MAX_LENGTH, TEXT_DIM, BASE_CHANNELS

def encode_text(texts, device):
    # UTF-8 byte tokenizer: deliberately simple, trainable embedding starts from scratch.
    ids = torch.zeros(len(texts), TEXT_MAX_LENGTH, dtype=torch.long)
    for i, text in enumerate(texts):
        raw = text.encode("utf-8")[:TEXT_MAX_LENGTH]
        if raw:
            ids[i, :len(raw)] = torch.tensor([b + 1 for b in raw])
    return ids.to(device)

class ResBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(cond_dim, channels * 2)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.film(cond).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.act(h))
        return x + h

class TinyTextDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(TEXT_VOCAB_SIZE, TEXT_DIM)
        self.text_mlp = nn.Sequential(nn.Linear(TEXT_DIM, TEXT_DIM), nn.SiLU(), nn.Linear(TEXT_DIM, TEXT_DIM))
        self.time_mlp = nn.Sequential(nn.Linear(1, TEXT_DIM), nn.SiLU(), nn.Linear(TEXT_DIM, TEXT_DIM))
        c = BASE_CHANNELS
        self.input = nn.Conv2d(3, c, 3, padding=1)
        self.block1 = ResBlock(c, TEXT_DIM)
        self.down = nn.Conv2d(c, c*2, 4, stride=2, padding=1)
        self.block2 = ResBlock(c*2, TEXT_DIM)
        self.mid = ResBlock(c*2, TEXT_DIM)
        self.up = nn.ConvTranspose2d(c*2, c, 4, stride=2, padding=1)
        self.block3 = ResBlock(c, TEXT_DIM)
        self.output = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, noisy, timesteps, tokens):
        text = self.token_embedding(tokens).mean(dim=1)
        t = timesteps.float().unsqueeze(1) / 200.0
        cond = self.text_mlp(text) + self.time_mlp(t)
        x1 = self.block1(self.input(noisy), cond)
        x2 = self.block2(self.down(x1), cond)
        x = self.mid(x2, cond)
        x = self.block3(self.up(x), cond)
        return self.output(x)
