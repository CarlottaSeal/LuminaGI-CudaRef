"""train.py <data_dir> [--epochs N] [--batch N] [--out path.pt]"""

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG16_Weights, vgg16


class VGGPerceptual(nn.Module):
    # L1 between VGG16 features at relu1_2 / relu2_2 / relu3_3.
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.s1 = vgg[:4]
        self.s2 = vgg[4:9]
        self.s3 = vgg[9:16]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        p = (pred   - self.mean) / self.std
        t = (target - self.mean) / self.std
        f1p = self.s1(p);  f1t = self.s1(t)
        f2p = self.s2(f1p); f2t = self.s2(f1t)
        f3p = self.s3(f2p); f3t = self.s3(f2t)
        return F.l1_loss(f1p, f1t) + F.l1_loss(f2p, f2t) + F.l1_loss(f3p, f3t)


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
    def __init__(self, base=32, in_ch=3):
        super().__init__()
        self.in_ch = in_ch
        self.d1 = ConvBlock(in_ch, base)
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
        # residual on x[:, :3] only — aux G-buffer channels are inputs, not targets.
        return torch.clamp(x[:, :3] + self.tail(u1), 0.0, 1.0)


class DenoiseDataset(Dataset):
    def __init__(self, data_dir, pair_indices, crop=256, gbuffer=False):
        d = pathlib.Path(data_dir)
        all_names = sorted(f.stem for f in (d / "noisy").glob("*.png"))
        self.pairs = []
        for stem in all_names:
            noisy = d / "noisy" / f"{stem}.png"
            clean = d / "clean" / f"{stem}.png"
            entry = {"noisy": noisy, "clean": clean}
            if gbuffer:
                entry["albedo"] = d / "clean" / f"{stem}.albedo.png"
                entry["normal"] = d / "clean" / f"{stem}.normal.png"
                entry["depth"]  = d / "clean" / f"{stem}.depth.png"
            self.pairs.append(entry)
        self.pairs = [self.pairs[i] for i in pair_indices]
        self.crop = crop
        self.gbuffer = gbuffer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        e = self.pairs[i]
        noisy = np.asarray(Image.open(e["noisy"]).convert("RGB"), dtype=np.float32) / 255.0
        clean = np.asarray(Image.open(e["clean"]).convert("RGB"), dtype=np.float32) / 255.0
        H, W = noisy.shape[:2]
        cropping = self.crop and self.crop < min(H, W)
        if cropping:
            x = np.random.randint(0, W - self.crop + 1)
            y = np.random.randint(0, H - self.crop + 1)
            noisy = noisy[y:y + self.crop, x:x + self.crop]
            clean = clean[y:y + self.crop, x:x + self.crop]

        if self.gbuffer:
            alb = np.asarray(Image.open(e["albedo"]).convert("RGB"), dtype=np.float32) / 255.0
            nrm = np.asarray(Image.open(e["normal"]).convert("RGB"), dtype=np.float32) / 255.0
            dep = np.asarray(Image.open(e["depth"]).convert("L"),   dtype=np.float32) / 255.0
            if cropping:
                alb = alb[y:y + self.crop, x:x + self.crop]
                nrm = nrm[y:y + self.crop, x:x + self.crop]
                dep = dep[y:y + self.crop, x:x + self.crop]
            stacked = np.concatenate([noisy, alb, nrm, dep[..., None]], axis=2)
            return (torch.from_numpy(stacked.transpose(2, 0, 1)),
                    torch.from_numpy(clean.transpose(2, 0, 1)))

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
    ap.add_argument("--perceptual-weight", type=float, default=0.0,
                    help="weight on VGG perceptual loss (0 = pure L1, default)")
    ap.add_argument("--gbuffer", action="store_true",
                    help="feed albedo+normal+depth alongside noisy RGB (10ch input)")
    args = ap.parse_args()
    in_ch = 10 if args.gbuffer else 3

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

    train_ds = DenoiseDataset(args.data_dir, train_idx, args.crop, gbuffer=args.gbuffer)
    val_ds   = DenoiseDataset(args.data_dir, val_idx,   args.crop, gbuffer=args.gbuffer)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet(base=args.base, in_ch=in_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f} M  in_ch={in_ch}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    perceptual = None
    if args.perceptual_weight > 0:
        perceptual = VGGPerceptual().to(device)
        print(f"perceptual: VGG16 relu1_2/relu2_2/relu3_3, weight={args.perceptual_weight}")
    else:
        print("perceptual: off (pure L1)")

    best_val = float("inf")
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_l1 = 0.0
        train_perc = 0.0
        for noisy, clean in train_dl:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            l1 = F.l1_loss(pred, clean)
            if perceptual is not None:
                perc = perceptual(pred, clean)
                loss = l1 + args.perceptual_weight * perc
                train_perc += perc.item() * noisy.size(0)
            else:
                loss = l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_l1 += l1.item() * noisy.size(0)
        train_l1 /= len(train_ds)
        train_perc /= len(train_ds)

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
            torch.save({"model": model.state_dict(), "base": args.base, "in_ch": in_ch}, out_path)
            tag = "  *"
        if perceptual is not None:
            print(f"epoch {epoch + 1:3d}/{args.epochs}  train_l1={train_l1:.4f}  "
                  f"train_perc={train_perc:.4f}  val_l1={val_loss:.4f}{tag}")
        else:
            print(f"epoch {epoch + 1:3d}/{args.epochs}  train={train_l1:.4f}  "
                  f"val={val_loss:.4f}{tag}")

    print(f"best val L1: {best_val:.4f}  saved: {out_path}")


if __name__ == "__main__":
    main()
