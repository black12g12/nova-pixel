import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import IMAGE_SIZE, TIMESTEPS
from dataset import ImageCaptionDataset
from model import TinyTextDenoiser, encode_text
from diffusion import Diffusion

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ImageCaptionDataset(args.data_dir, IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device=="cuda"))
    model = TinyTextDenoiser().to(device)
    diffusion = Diffusion(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    Path("checkpoints").mkdir(exist_ok=True)
    print(f"Device: {device} | images: {len(ds)} | parameters: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, captions in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            tokens = encode_text(list(captions), device)
            t = torch.randint(0, TIMESTEPS, (images.size(0),), device=device)
            noisy, noise = diffusion.add_noise(images, t)
            predicted = model(noisy, t, tokens)
            loss = nn.functional.mse_loss(predicted, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / max(1, len(loader))
        print(f"Epoch {epoch+1}: loss={avg:.5f}")
        torch.save({"model": model.state_dict(), "epoch": epoch+1}, "checkpoints/latest.pt")

if __name__ == "__main__":
    main()
