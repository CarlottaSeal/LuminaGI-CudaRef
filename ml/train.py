"""train.py <data_dir> [--epochs N] [--batch N] [--out path.pt]

Simple U-Net denoiser. Crop-based augmentation, L1 loss, Adam.
"""

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.d1 = ConvBlock(3, base)
        self.d2 = ConvBlock(base, base * 2)
        self.d3 = ConvBlock(base * 2, base * 4)
        self.d4 = ConvBlock(base * 4, base * 8)
        self.u3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.u2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.u1 = ConvBlock(base * 2 + base,     base)
        self.tail = nn.Conv2d(base, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        u3 = self.u3(torch.cat([self.up(d4), d3], 1))
        u2 = self.u2(torch.cat([self.up(u3), d2], 1))
        u1 = self.u1(torch.cat([self.up(u2), d1], 1))
        # Predict a residual on top of the noisy input, clamped to [0,1].
        return torch.clamp(x + self.tail(u1), 0.0, 1.0)


class DenoiseDataset(Dataset):
    def __init__(self, data_dir, pair_indices, crop=256):
        d = pathlib.Path(data_dir)
        all_names = sorted(f.name for f in (d / "noisy").glob("*.png"))
        self.pairs = [(d / "noisy" / n, d / "clean" / n) for n in all_names]
        self.pairs = [self.pairs[i] for i in pair_indices]
        self.crop = crop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        noisy_p, clean_p = self.pairs[i]
        noisy = np.asarray(Image.open(noisy_p).convert("RGB"), dtype=np.float32) / 255.0
        clean = np.asarray(Image.open(clean_p).convert("RGB"), dtype=np.float32) / 255.0
        H, W = noisy.shape[:2]
        if self.crop and self.crop < min(H, W):
            x = np.random.randint(0, W - self.crop + 1)
            y = np.random.randint(0, H - self.crop + 1)
            noisy = noisy[y:y + self.crop, x:x + self.crop]
            clean = clean[y:y + self.crop, x:x + self.crop]
        return (torch.from_numpy(noisy.transpose(2, 0, 1)),
                torch.from_numpy(clean.transpose(2, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--epochs",   type=int,   default=80)
    ap.add_argument("--batch",    type=int,   default=4)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--crop",     type=int,   default=256)
    ap.add_argument("--base",     type=int,   default=32)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out",      default="ml/runs/denoiser.pt")
    ap.add_argument("--seed",     type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    d = pathlib.Path(args.data_dir)
    n_total = len(list((d / "noisy").glob("*.png")))
    n_val = max(1, int(n_total * args.val_frac))
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_total)
    train_idx = perm[n_val:].tolist()
    val_idx   = perm[:n_val].tolist()

    train_ds = DenoiseDataset(args.data_dir, train_idx, args.crop)
    val_ds   = DenoiseDataset(args.data_dir, val_idx,   args.crop)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet(base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f} M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for noisy, clean in train_dl:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            loss = F.l1_loss(pred, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * noisy.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                val_loss += F.l1_loss(model(noisy), clean).item() * noisy.size(0)
        val_loss /= len(val_ds)

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "base": args.base}, out_path)
            tag = "  *"
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}{tag}")

    print(f"best val L1: {best_val:.4f}  saved: {out_path}")


if __name__ == "__main__":
    main()
