"""
train.py — Reproduce Training Process from Scratch for SemiconDaAIR-v5.

Usage:
  python train.py --data_dir /path/to/dataset --epochs 25 --batch_size 16 --lr 2e-4
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.semicon_daair_v5 import build_semicon_daair_v5
from dataset import RealPairedSemiconductorDataset
from utils.metrics import compute_psnr, compute_ssim


class CharbonnierEdgeLoss(nn.Module):
    def __init__(self, eps=1e-3, alpha=0.2):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, gt):
        diff = pred - gt
        loss_char = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        px = torch.nn.functional.conv2d(pred, self.sobel_x, padding=1)
        py = torch.nn.functional.conv2d(pred, self.sobel_y, padding=1)
        gx = torch.nn.functional.conv2d(gt, self.sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(gt, self.sobel_y, padding=1)
        loss_edge = torch.mean(torch.abs(torch.sqrt(px**2 + py**2 + 1e-6) - torch.sqrt(gx**2 + gy**2 + 1e-6)))
        return loss_char + self.alpha * loss_edge


def main():
    parser = argparse.ArgumentParser(description="SemiconDaAIR-v5 Training Script")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\HP\Downloads\dataset\train\train", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints/final", help="Checkpoint directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print("      SEMICONDAAIR-V5 TRAINING PIPELINE REPRODUCIBILITY      ")
    print("=" * 70)
    print(f"Data Directory : {args.data_dir}")
    print(f"Total Epochs   : {args.epochs}")
    print(f"Batch Size     : {args.batch_size}")
    print(f"Learning Rate  : {args.lr}")
    print(f"Device         : {device}")

    train_ds = RealPairedSemiconductorDataset(args.data_dir, split_file="splits/train.txt")
    val_ds = RealPairedSemiconductorDataset(args.data_dir, split_file="splits/val.txt")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_semicon_daair_v5(scale=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CharbonnierEdgeLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for lq, gt in train_loader:
            lq, gt = lq.to(device), gt.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(lq)
                loss = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for lq, gt in val_loader:
                lq, gt = lq.to(device), gt.to(device)
                pred = model(lq)
                p_np = pred.squeeze().cpu().numpy()
                g_np = gt.squeeze().cpu().numpy()
                psnr_list.append(compute_psnr(p_np, g_np))
                ssim_list.append(compute_ssim(p_np, g_np))

        val_psnr = float(torch.tensor(psnr_list).mean().item())
        val_ssim = float(torch.tensor(ssim_list).mean().item())

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {avg_loss:.5f} | Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt_path = os.path.join(args.save_dir, "semicon_daair_v5_candidate.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim
            }, ckpt_path)
            print(f"  [SAVED BEST V5] {ckpt_path} (PSNR: {val_psnr:.4f} dB)")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
