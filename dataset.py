import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 64):
        self.root = Path(data_dir)
        csv_path = self.root / "captions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}. Create captions.csv with filename,caption columns.")
        with csv_path.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows or not {"filename", "caption"}.issubset(self.rows[0]):
            raise ValueError("captions.csv must contain filename,caption columns.")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.root / "images" / row["filename"]).convert("RGB")
        return self.transform(image), row["caption"]
