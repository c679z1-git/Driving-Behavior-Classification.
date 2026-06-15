import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

CLASSES     = ["normal", "swerving", "tailgating"]
NUM_CLASSES = len(CLASSES)
NUM_FRAMES  = 32
IMG_SIZE    = 112
VAL_RATIO   = 0.2

m1 = [0.485, 0.456, 0.406]
s1 = [0.229, 0.224, 0.225]

aug_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomGrayscale(p=0.05),
])
aug_val = T.Compose([])


def _sample_indices(total: int, n: int) -> list:
    if total <= n:
        return list(range(total)) + [total - 1] * (n - total)
    st = total / n
    return [int(i * st) for i in range(n)]


class ClipDataset(Dataset):
    def __init__(self, root: str, split: str = "train", max_clips: int = 70):
        self.root  = Path(root)
        self.split = split
        self.aug   = aug_train if split == "train" else aug_val
        self.norm  = T.Compose([T.ToTensor(), T.Normalize(m1, s1)])
        self.samples: list[tuple[Path, int]] = []

        for li, cls in enumerate(CLASSES):
            cd = self.root / cls
            if not cd.exists():
                print(f"  [WARN] class folder not found: {cd}")
                continue
            cc = sorted(p for p in cd.iterdir() if p.is_dir())
            if not cc:
                print(f"  [WARN] no clip subfolders in: {cd}")
                continue
            random.shuffle(cc)
            cc  = cc[:max_clips]
            cut = max(1, int(len(cc) * (1 - VAL_RATIO)))
            pp  = cc[:cut] if split == "train" else cc[cut:]
            for c in pp:
                self.samples.append((c, li))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _load_clip(self, cd: Path) -> torch.Tensor:
        fp = sorted(
            p for p in cd.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not fp:
            return torch.zeros(NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)
        idx = _sample_indices(len(fp), NUM_FRAMES)
        ff  = []
        for i in idx:
            img = Image.open(fp[i]).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img = self.aug(img)
            ff.append(self.norm(img))
        return torch.stack(ff)

    def __getitem__(self, idx):
        cd, lbl = self.samples[idx]
        return self._load_clip(cd), lbl

    def class_weights(self) -> torch.Tensor:
        cnt = torch.zeros(NUM_CLASSES)
        for _, lbl in self.samples:
            cnt[lbl] += 1
        cnt = cnt.clamp(min=1)
        w   = 1.0 / cnt
        return torch.tensor([w[lbl] for _, lbl in self.samples])
