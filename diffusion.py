import torch
from config import TIMESTEPS

class Diffusion:
    def __init__(self, device):
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, images, t, noise=None):
        if noise is None:
            noise = torch.randn_like(images)
        a = self.alpha_bars[t].view(-1, 1, 1, 1)
        return a.sqrt() * images + (1 - a).sqrt() * noise, noise

    @torch.no_grad()
    def sample(self, model, tokens, image_size, channels=3):
        model.eval()
        x = torch.randn(tokens.size(0), channels, image_size, image_size, device=self.device)
        for step in reversed(range(TIMESTEPS)):
            t = torch.full((x.size(0),), step, device=self.device, dtype=torch.long)
            eps = model(x, t, tokens)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            x = (x - beta / torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha)
            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x.clamp(-1, 1)
