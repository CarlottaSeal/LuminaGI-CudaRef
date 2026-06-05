"""train.py <data_dir> [--epochs N] [--batch N] [--out path.pt]"""

import argparse
import copy
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG16_Weights, vgg16

from ecb import ECB


class EMA:
    # exponential moving average of model parameters; eval the EMA copy, train the raw model.
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for p_s, p_m in zip(self.shadow.parameters(), model.parameters()):
            p_s.mul_(self.decay).add_(p_m.detach(), alpha=1.0 - self.decay)
        for b_s, b_m in zip(self.shadow.buffers(), model.buffers()):
            b_s.copy_(b_m)


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
    def __init__(self, in_ch, out_ch, conv_type="plain"):
        super().__init__()
        mk = (lambda i, o: ECB(i, o)) if conv_type == "ecb" \
            else (lambda i, o: nn.Conv2d(i, o, 3, padding=1))
        self.net = nn.Sequential(
            mk(in_ch,  out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            mk(out_ch, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=32, in_ch=3, conv_type="plain", scale=1):
        super().__init__()
        self.in_ch = in_ch
        self.scale = scale
        self.d1 = ConvBlock(in_ch, base, conv_type)
        self.d2 = ConvBlock(base, base * 2, conv_type)
        self.d3 = ConvBlock(base * 2, base * 4, conv_type)
        self.d4 = ConvBlock(base * 4, base * 8, conv_type)
        self.u3 = ConvBlock(base * 8 + base * 4, base * 4, conv_type)
        self.u2 = ConvBlock(base * 4 + base * 2, base * 2, conv_type)
        self.u1 = ConvBlock(base * 2 + base,     base, conv_type)
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")
        if scale == 1:
            self.tail = nn.Conv2d(base, 3, 1)  # 1x1 head stays plain; ECB only folds 3x3
        else:
            # sub-pixel SR head (ECBSR style): predict scale^2 * 3 channels then shuffle up
            self.tail = nn.Sequential(
                nn.Conv2d(base, 3 * scale * scale, 3, padding=1),
                nn.PixelShuffle(scale),
            )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        u3 = self.u3(torch.cat([self.up(d4), d3], 1))
        u2 = self.u2(torch.cat([self.up(u3), d2], 1))
        u1 = self.u1(torch.cat([self.up(u2), d1], 1))
        if self.scale == 1:
            # residual on x[:, :3] only — aux G-buffer channels are inputs, not targets.
            base_img = x[:, :3]
        else:
            # learn the high-frequency detail on top of a bilinear upscale of the input
            base_img = F.interpolate(x[:, :3], scale_factor=self.scale,
                                     mode="bilinear", align_corners=False)
        return torch.clamp(base_img + self.tail(u1), 0.0, 1.0)


class DenoiseDataset(Dataset):
    def __init__(self, data_dir, pair_indices, crop=256, gbuffer=False, scale=1):
        if scale > 1 and gbuffer:
            raise ValueError("SR (scale>1) + gbuffer not supported together yet")
        d = pathlib.Path(data_dir)
        # only main color frames: stem is all digits (skip *.albedo.png / *.normal.png / *.worldpos.png)
        all_names = sorted(f.stem for f in (d / "noisy").glob("*.png") if f.stem.isdigit())
        self.pairs = []
        for stem in all_names:
            noisy = d / "noisy" / f"{stem}.png"
            clean = d / "clean" / f"{stem}.png"
            entry = {"noisy": noisy, "clean": clean}
            if gbuffer:
                entry["albedo"]   = d / "clean" / f"{stem}.albedo.png"
                entry["normal"]   = d / "clean" / f"{stem}.normal.png"
                entry["worldpos"] = d / "clean" / f"{stem}.worldpos.png"
            self.pairs.append(entry)
        self.pairs = [self.pairs[i] for i in pair_indices]
        self.crop = crop
        self.gbuffer = gbuffer
        self.scale = scale

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
            alb = np.asarray(Image.open(e["albedo"]).convert("RGB"),   dtype=np.float32) / 255.0
            nrm = np.asarray(Image.open(e["normal"]).convert("RGB"),   dtype=np.float32) / 255.0
            wp  = np.asarray(Image.open(e["worldpos"]).convert("RGB"), dtype=np.float32) / 255.0
            if cropping:
                alb = alb[y:y + self.crop, x:x + self.crop]
                nrm = nrm[y:y + self.crop, x:x + self.crop]
                wp  = wp [y:y + self.crop, x:x + self.crop]
            stacked = np.concatenate([noisy, alb, nrm, wp], axis=2)
            return (torch.from_numpy(stacked.transpose(2, 0, 1)),
                    torch.from_numpy(clean.transpose(2, 0, 1)))

        if self.scale > 1:
            # joint denoise+SR pair: clean crop is the HR target; the LR input is the
            # noisy crop bicubic-downscaled by `scale` (the DIV2K / ECBSR protocol).
            clean_hr = torch.from_numpy(clean.transpose(2, 0, 1))
            n = torch.from_numpy(noisy.transpose(2, 0, 1)).unsqueeze(0)
            h, w = n.shape[2] // self.scale, n.shape[3] // self.scale
            low = F.interpolate(n, size=(h, w), mode="bicubic", align_corners=False)
            return low.clamp_(0.0, 1.0).squeeze(0), clean_hr

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
                    help="feed albedo+normal+worldpos alongside noisy RGB (12ch input)")
    ap.add_argument("--ecb", action="store_true",
                    help="use edge-oriented conv blocks (ECBSR) in place of plain 3x3 convs; "
                         "folds to a single 3x3 at eval/deploy, so no inference cost change")
    ap.add_argument("--scale", type=int, default=1,
                    help="super-resolution factor: 1 = denoise only; >1 = joint denoise+SR "
                         "(LR input = clean-target's noisy crop bicubic-downscaled by scale)")
    ap.add_argument("--optim", choices=["adam", "adamw_cosine"], default="adam",
                    help="adam = vanilla baseline; adamw_cosine = AdamW + cosine LR + grad clip + EMA")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="AdamW weight decay (only used with --optim adamw_cosine)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="clip_grad_norm_ max_norm (only used with --optim adamw_cosine; 0 disables)")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA decay (only used with --optim adamw_cosine; 0 disables EMA)")
    args = ap.parse_args()
    in_ch = 12 if args.gbuffer else 3
    if args.scale > 1 and args.crop % args.scale:
        sys.exit(f"--crop ({args.crop}) must be divisible by --scale ({args.scale})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    d = pathlib.Path(args.data_dir)
    n_total = sum(1 for f in (d / "noisy").glob("*.png") if f.stem.isdigit())
    n_val = max(1, int(n_total * args.val_frac))
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_total)
    train_idx = perm[n_val:].tolist()
    val_idx   = perm[:n_val].tolist()

    train_ds = DenoiseDataset(args.data_dir, train_idx, args.crop, gbuffer=args.gbuffer, scale=args.scale)
    val_ds   = DenoiseDataset(args.data_dir, val_idx,   args.crop, gbuffer=args.gbuffer, scale=args.scale)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    conv_type = "ecb" if args.ecb else "plain"
    model = UNet(base=args.base, in_ch=in_ch, conv_type=conv_type, scale=args.scale).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    task = "denoise" if args.scale == 1 else f"denoise+SR x{args.scale}"
    print(f"params: {n_params/1e6:.2f} M  in_ch={in_ch}  conv={conv_type}  task={task}"
          + ("  (multi-branch at train time, folds to single 3x3 at eval)" if args.ecb else ""))

    if args.optim == "adamw_cosine":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
        print(f"optim: AdamW(lr={args.lr}, wd={args.weight_decay}) + cosine T_max={args.epochs} "
              f"+ grad_clip={args.grad_clip} + ema_decay={args.ema_decay}")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = None
        ema = None
        print(f"optim: Adam(lr={args.lr})  (vanilla baseline)")

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
            if args.optim == "adamw_cosine" and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(model)
            train_l1 += l1.item() * noisy.size(0)
        train_l1 /= len(train_ds)
        train_perc /= len(train_ds)
        if sched is not None:
            sched.step()

        eval_model = ema.shadow if ema is not None else model
        eval_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                val_loss += F.l1_loss(eval_model(noisy), clean).item() * noisy.size(0)
        val_loss /= len(val_ds)

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            save_state = eval_model.state_dict()
            torch.save({"model": save_state, "base": args.base, "in_ch": in_ch,
                        "conv_type": conv_type, "scale": args.scale}, out_path)
            tag = "  *"
        lr_now = opt.param_groups[0]["lr"]
        if perceptual is not None:
            print(f"epoch {epoch + 1:3d}/{args.epochs}  lr={lr_now:.2e}  train_l1={train_l1:.4f}  "
                  f"train_perc={train_perc:.4f}  val_l1={val_loss:.4f}{tag}")
        else:
            print(f"epoch {epoch + 1:3d}/{args.epochs}  lr={lr_now:.2e}  train={train_l1:.4f}  "
                  f"val={val_loss:.4f}{tag}")

    print(f"best val L1: {best_val:.4f}  saved: {out_path}")


if __name__ == "__main__":
    main()
