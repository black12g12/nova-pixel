import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from config import IMAGE_SIZE
from model import TinyTextDenoiser, encode_text
from diffusion import Diffusion
from rules import check_prompt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="outputs/generated.png")
    args = p.parse_args()

    allowed, reason = check_prompt(args.prompt)
    if not allowed:
        raise SystemExit(reason)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTextDenoiser().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    tokens = encode_text([args.prompt], device)
    image = Diffusion(device).sample(model, tokens, IMAGE_SIZE)
    image = (image + 1) / 2
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_image(image, args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
