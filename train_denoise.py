import os, math, argparse, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image as tv_save

from pytorch_msssim import ms_ssim, ssim


# ------------------ 基础 ------------------
def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def psnr_torch(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(x, y, reduction='mean').item()
    if mse == 0: return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)

def ssim_torch(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    return ssim(x, y, data_range=data_range, size_average=True).item()


# ------------------ 数据 ------------------
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def scan_images(root: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, f))
    out.sort()
    return out

def build_pairs(noisy_root: str, clean_root: str) -> List[Tuple[str, str]]:
    noisy_list = scan_images(noisy_root)
    pairs = []
    for n in noisy_list:
        rel = os.path.relpath(n, noisy_root)
        c = os.path.join(clean_root, rel)
        if os.path.exists(c):
            pairs.append((n, c))
    return pairs

# ------------------ Dataset（保证成对样本尺寸一致） ------------------
class PairPatchDataset(Dataset):
    """
    - train=True：按两幅图的公共尺寸随机对齐裁块（相对位置一致），并做随机翻转
    - train=False：按两幅图的公共尺寸做中心裁切，保证 noisy / clean 尺寸完全一致
    - 若 --resize>0，则先统一 resize，再进入裁切逻辑
    """
    def __init__(self, pairs: List[Tuple[str,str]], crop: int=256, resize: int=0, train: bool=True, channels:int=1):
        super().__init__()
        self.pairs = pairs
        self.crop = int(crop)
        self.resize = int(resize)
        self.train = train
        self.channels = channels

        tfms = []
        if self.resize and self.resize > 0:
            tfms.append(T.Resize((self.resize, self.resize), interpolation=T.InterpolationMode.BILINEAR))
        if channels == 1:
            tfms.append(T.Grayscale(num_output_channels=1))
        tfms.append(T.ToTensor())
        self.base = T.Compose(tfms)

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        top  = max(0, (H - h) // 2)
        left = max(0, (W - w) // 2)
        return x[..., top:top+h, left:left+w]

    def __getitem__(self, idx):
        n_path, c_path = self.pairs[idx]
        n = self.base(Image.open(n_path).convert('RGB'))
        c = self.base(Image.open(c_path).convert('RGB'))

        Hn, Wn = n.shape[-2], n.shape[-1]
        Hc, Wc = c.shape[-2], c.shape[-1]
        Hcom, Wcom = min(Hn, Hc), min(Wn, Wc)         # 两图公共可裁尺寸

        # 目标 patch 尺寸
        if self.crop and self.crop > 0:
            ph = min(self.crop, Hcom)
            pw = min(self.crop, Wcom)
        else:
            ph, pw = Hcom, Wcom

        if self.train:
            # 随机但“相对一致”的位置：用同一个 u,v 生成两图的起点（按各自尺寸换算）
            import random
            u = random.random(); v = random.random()
            ta = int(u * max(0, Hn - ph)); la = int(v * max(0, Wn - pw))
            tb = int(u * max(0, Hc - ph)); lb = int(v * max(0, Wc - pw))
            n = n[..., ta:ta+ph, la:la+pw]
            c = c[..., tb:tb+ph, lb:lb+pw]

            # 随机翻转
            if random.random() < 0.5:
                n = torch.flip(n, dims=[2]); c = torch.flip(c, dims=[2])
            if random.random() < 0.5:
                n = torch.flip(n, dims=[1]); c = torch.flip(c, dims=[1])
        else:
            # 验证/测试：中心对齐裁切到相同尺寸
            n = self._center_crop(n, ph, pw)
            c = self._center_crop(c, ph, pw)

        # 现在 n、c 的尺寸严格一致
        return n, c


# ------------------ 模型 ------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        self.e1 = DoubleConv(in_ch, base); self.p1 = nn.MaxPool2d(2)
        self.e2 = DoubleConv(base, base*2); self.p2 = nn.MaxPool2d(2)
        self.e3 = DoubleConv(base*2, base*4); self.p3 = nn.MaxPool2d(2)
        self.mid = DoubleConv(base*4, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2); self.d3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2); self.d2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2);   self.d1 = DoubleConv(base*2, base)
        self.out_conv = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        m  = self.mid(self.p3(e3))
        u3 = self.u3(m); e3 = F.interpolate(e3, size=u3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.d3(torch.cat([u3, e3], 1))
        u2 = self.u2(d3); e2 = F.interpolate(e2, size=u2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], 1))
        u1 = self.u1(d2); e1 = F.interpolate(e1, size=u1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], 1))
        return torch.clamp(self.out_conv(d1), 0, 1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.net(x) + x)

class ResUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True), ResBlock(base))
        self.p1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(inplace=True), ResBlock(base*2))
        self.p2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(inplace=True), ResBlock(base*4))
        self.p3 = nn.MaxPool2d(2)
        self.mid  = nn.Sequential(nn.Conv2d(base*4, base*8, 3, padding=1), nn.ReLU(inplace=True), ResBlock(base*8))
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = nn.Sequential(ResBlock(base*8), nn.Conv2d(base*8, base*4, 3, padding=1), nn.ReLU(inplace=True))
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = nn.Sequential(ResBlock(base*4), nn.Conv2d(base*4, base*2, 3, padding=1), nn.ReLU(inplace=True))
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = nn.Sequential(ResBlock(base*2), nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.p1(e1))
        e3 = self.enc3(self.p2(e2))
        m  = self.mid(self.p3(e3))
        u3 = self.u3(m); e3 = F.interpolate(e3, size=u3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], 1))
        u2 = self.u2(d3); e2 = F.interpolate(e2, size=u2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], 1))
        u1 = self.u1(d2); e1 = F.interpolate(e1, size=u1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], 1))
        return torch.clamp(self.out_conv(d1), 0, 1)

# ---- 简化版 Swin（仅 Window-MSA，无 shift，轻量） ----
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

def window_partition(x, win):  # x:[B,C,H,W] -> [B*nw,C,win,win]
    B, C, H, W = x.shape
    x = x.view(B, C, H//win, win, W//win, win)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, win, win)
    return x

def window_reverse(windows, win, H, W, B):  # windows:[B*nw,C,win,win] -> [B,C,H,W]
    C = windows.shape[1]
    x = windows.view(B, H//win, W//win, C, win, win)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x

class WindowMSA(nn.Module):
    def __init__(self, dim, num_heads=4, win=8):
        super().__init__()
        self.win = win
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):  # x:[B,C,H,W]
        B, C, H, W = x.shape
        assert H % self.win == 0 and W % self.win == 0, "H/W 必须可被 window 整除"
        ws = window_partition(x, self.win)                 # [B*nw,C,win,win]
        ws = ws.flatten(2).transpose(1, 2)                 # [B*nw, L, C]
        ws = self.norm(ws)
        out, _ = self.attn(ws, ws, ws)                     # [B*nw, L, C]
        out = self.proj(out).transpose(1, 2)               # [B*nw, C, L]
        out = out.view(-1, C, self.win, self.win)
        return window_reverse(out, self.win, H, W, B)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, win=8, mlp_ratio=4):
        super().__init__()
        self.msa = WindowMSA(dim, num_heads=num_heads, win=win)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.n1 = nn.BatchNorm2d(dim); self.n2 = nn.BatchNorm2d(dim)
    def forward(self, x):
        x = x + self.msa(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class SwinStage(nn.Module):
    def __init__(self, dim, depth=2, heads=4, win=8):
        super().__init__()
        self.blocks = nn.Sequential(*[SwinBlock(dim, heads, win=win) for _ in range(depth)])
    def forward(self, x): return self.blocks(x)

# ---- 模型 1：swin-only（单尺度；stem→多层Swin→tail） ----
class SwinOnly(nn.Module):
    """
    纯 Swin 窗口注意力去噪（无 U-Net 跳连/金字塔），保持分辨率不变。
    建议 crop_size / win 取整（默认 win=8；crop=256 OK）。
    """
    def __init__(self, in_ch=1, out_ch=1, base=96, heads=8, depth=6, win=8):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.body = SwinStage(base, depth=depth, heads=heads, win=win)
        self.tail = nn.Conv2d(base, out_ch, 3, padding=1)
    def forward(self, x):
        y = self.stem(x)
        y = self.body(y)
        y = self.tail(y)
        return torch.clamp(y, 0, 1)

# ---- 模型 2：ST-ResUNet（ResUNet 主干 + bottleneck 接入 Swin） ----
class STResUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, swin_heads=8, swin_depth=2, win=8):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True), ResBlock(base))
        self.p1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(inplace=True), ResBlock(base*2))
        self.p2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(inplace=True), ResBlock(base*4))
        self.p3 = nn.MaxPool2d(2)
        self.mid_pre  = nn.Conv2d(base*4, base*8, 3, padding=1)
        self.swin     = SwinStage(base*8, depth=swin_depth, heads=swin_heads, win=win)
        self.mid_post = nn.Conv2d(base*8, base*8, 3, padding=1)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = nn.Sequential(ResBlock(base*8), nn.Conv2d(base*8, base*4, 3, padding=1), nn.ReLU(inplace=True))
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = nn.Sequential(ResBlock(base*4), nn.Conv2d(base*4, base*2, 3, padding=1), nn.ReLU(inplace=True))
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = nn.Sequential(ResBlock(base*2), nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.p1(e1))
        e3 = self.enc3(self.p2(e2))
        m  = self.mid_post(self.swin(self.mid_pre(self.p3(e3))))
        u3 = self.u3(m); e3 = F.interpolate(e3, size=u3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], 1))
        u2 = self.u2(d3); e2 = F.interpolate(e2, size=u2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], 1))
        u1 = self.u1(d2); e1 = F.interpolate(e1, size=u1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], 1))
        return torch.clamp(self.out_conv(d1), 0, 1)


# ------------------ 损失 ------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3): super().__init__(); self.eps = eps
    def forward(self, x, y): return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))

class HybridLoss(nn.Module):
    def __init__(self, mode='charb_ms', ssim_w=0.1):
        super().__init__()
        self.mode = mode; self.ssim_w = ssim_w
        self.charb = CharbonnierLoss(); self.l1 = nn.L1Loss()
    def forward(self, out, tgt):
        if 'charb' in self.mode: loss_pix = self.charb(out, tgt)
        else:                     loss_pix = self.l1(out, tgt)
        if 'ms' in self.mode:
            return (1.0 - self.ssim_w) * loss_pix + self.ssim_w * (1.0 - ms_ssim(out, tgt, data_range=1.0, size_average=True))
        return loss_pix


# ------------------ 训练/评估 ------------------
def _parse_heads_depths(s: str, default_len=3) -> List[int]:
    arr = [int(x) for x in str(s).split(',') if x != '']
    if len(arr) < default_len:
        arr += [arr[-1] if arr else 4] * (default_len - len(arr))
    return arr[:default_len]

def _build_model(name: str, in_ch: int, out_ch: int, width: int, swin_heads: List[int], swin_depths: List[int], win: int):
    name = name.lower()
    if name == 'swin_only':
        # 用 heads[0], depths[0] 作为该单尺度模型的 heads/depth
        return SwinOnly(in_ch, out_ch, base=width, heads=swin_heads[0], depth=swin_depths[0], win=win)
    if name == 'unet':
        return UNet(in_ch, out_ch, base=width)
    if name == 'resunet':
        return ResUNet(in_ch, out_ch, base=width)
    if name == 'stresunet':
        return STResUNet(in_ch, out_ch, base=width, swin_heads=swin_heads[-1], swin_depth=swin_depths[-1], win=win)
    raise ValueError(f"未知模型：{name}")

def _post_process(args, inp, net_out):
    if args.residual:
        return torch.clamp(inp + args.residual_scale * net_out, 0, 1)
    return net_out

def train_one(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(1234)

    pairs_all = build_pairs(args.train_noisy, args.train_clean)
    if len(pairs_all) == 0:
        raise FileNotFoundError("训练集为空或路径不匹配")
    random.shuffle(pairs_all)
    n_val = max(1, int(0.1 * len(pairs_all)))
    val_pairs = pairs_all[:n_val]; train_pairs = pairs_all[n_val:]

    # debug grid
    # debug grid（保证尺寸一致再拼图）
    debug_dir = Path(args.out_dir) / "debug_pairs";
    ensure_dir(debug_dir)
    # 统一成 crop_size（若未设则用 256），仅用于可视化，不影响训练数据流
    dbg_size = args.crop_size if args.crop_size > 0 else 256
    dbg_tfms = T.Compose(
        [T.Resize((dbg_size, dbg_size), interpolation=T.InterpolationMode.BILINEAR)]
        + ([T.Grayscale(num_output_channels=1)] if args.channels == 1 else [])
        + [T.ToTensor()]
    )
    grid_imgs = []
    for i in range(min(8, len(val_pairs))):
        n_path, c_path = val_pairs[i]
        n = dbg_tfms(Image.open(n_path).convert('RGB'))
        c = dbg_tfms(Image.open(c_path).convert('RGB'))
        grid_imgs.extend([n, c])
    if grid_imgs:
        grid = make_grid(torch.stack(grid_imgs, 0), nrow=2, padding=2)
        tv_save(grid, str(debug_dir / "pairs_grid.png"))
    print(f"✅ 写出数据配对自检图到：{debug_dir}")

    # baseline（val noisy vs clean）
    val_ds = PairPatchDataset(val_pairs, crop=args.crop_size, resize=args.resize, train=False, channels=args.channels)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    base_psnr = base_ssim = 0.0; cnt = 0
    for n, c in val_loader:
        n, c = n.to(device), c.to(device)
        base_psnr += psnr_torch(n, c); base_ssim += ssim_torch(n, c); cnt += 1
    print(f"🔎 Baseline(noisy vs clean on val): PSNR={base_psnr/max(1,cnt):.2f}, SSIM={base_ssim/max(1,cnt):.4f}")

    # train
    train_ds = PairPatchDataset(train_pairs, crop=args.crop_size, resize=args.resize, train=True, channels=args.channels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    heads = _parse_heads_depths(args.swin_heads)
    depths = _parse_heads_depths(args.swin_depths)
    model = _build_model(args.model, args.channels, args.channels, args.width, heads, depths, args.win).to(device)
    loss_fn = HybridLoss(args.loss, args.ssim_w)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_psnr, best_epoch, bad = -1.0, 0, 0
    ckpt_dir = Path(args.ckpt_dir); ensure_dir(ckpt_dir)
    out_vis = Path(args.out_dir); ensure_dir(out_vis)

    for epoch in range(1, args.epochs+1):
        model.train(); optim.zero_grad(set_to_none=True)
        for i, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            pred = _post_process(args, noisy, pred)
            loss = loss_fn(pred, clean) / max(1, args.accum_steps)
            loss.backward()
            if (i+1) % args.accum_steps == 0:
                optim.step(); optim.zero_grad(set_to_none=True)

        # val
        model.eval(); v_psnr = v_ssim = 0.0; v_cnt = 0
        with torch.no_grad():
            for n, c in val_loader:
                n, c = n.to(device), c.to(device)
                out = _post_process(args, n, model(n))
                v_psnr += psnr_torch(out, c); v_ssim += ssim_torch(out, c); v_cnt += 1
        v_psnr /= max(1, v_cnt); v_ssim /= max(1, v_cnt)
        print(f"Val @ Epoch {epoch}: PSNR={v_psnr:.2f}, SSIM={v_ssim:.4f}")

        if v_psnr > best_psnr:
            best_psnr, best_epoch, bad = v_psnr, epoch, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'psnr': v_psnr}, str(ckpt_dir / f'{args.model}_best.pth'))
        else:
            bad += 1
            if bad >= args.patience_hit:
                print(f"⏹️ 提前停止：连续 {args.patience_hit} 轮未提升。")
                break

        if epoch % 10 == 0:
            with torch.no_grad():
                n, _ = next(iter(val_loader))
                n = n.to(device); vis = _post_process(args, n, model(n))
                tv_save(make_grid(torch.cat([n, vis], 0), nrow=n.shape[0]), str(out_vis / f'epoch{epoch:03d}_val_vis.png'))

    best_path = ckpt_dir / f'{args.model}_best.pth'
    if best_path.exists():
        state = torch.load(str(best_path), map_location='cpu')
        model.load_state_dict(state['model'], strict=True)
        print(f"✅ 已载入最佳权重：epoch={state.get('epoch', -1)}  val_psnr={state.get('psnr', 0.0):.2f}")
    else:
        print("⚠️ 未找到最佳权重，使用最后一轮参数。")
    return model

def _tta_predict(model, x):
    outs = []
    outs.append(model(x))
    outs.append(torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1]))
    outs.append(torch.flip(model(torch.flip(x, dims=[-2])), dims=[-2]))
    outs.append(torch.flip(torch.flip(model(torch.flip(torch.flip(x, dims=[-1]), dims=[-2])), dims=[-2]), dims=[-1]))
    return torch.clamp(torch.stack(outs, 0).mean(0), 0, 1)

def evaluate(args, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        heads = _parse_heads_depths(args.swin_heads)
        depths = _parse_heads_depths(args.swin_depths)
        model = _build_model(args.model, args.channels, args.channels, args.width, heads, depths, args.win)
        ckpt = Path(args.ckpt_dir) / f'{args.model}_best.pth'
        if not ckpt.exists(): raise FileNotFoundError(f"找不到权重：{ckpt}")
        state = torch.load(str(ckpt), map_location='cpu')
        model.load_state_dict(state['model'], strict=False)
        print(f"✅ 已载入最佳权重：epoch={state.get('epoch', -1)}  val_psnr={state.get('psnr', 0.0):.2f}")
    model = model.to(device).eval()

    pairs = build_pairs(args.test_noisy, args.test_clean)
    if len(pairs) == 0:
        print("⚠️ 测试集为空或路径不匹配。"); return 0.0, 0.0
    test_ds = PairPatchDataset(pairs, crop=args.crop_size, resize=args.resize, train=False, channels=args.channels)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    out_dir = Path(args.out_dir) / "test_outputs"; ensure_dir(out_dir)
    psnr_sum = ssim_sum = 0.0; cnt = 0
    with torch.no_grad():
        for i, (n, c) in enumerate(test_loader):
            n, c = n.to(device), c.to(device)
            o = _tta_predict(model, n) if args.tta else model(n)
            o = _post_process(args, n, o)
            tv_save(o, str(out_dir / f'{i:05d}.png'))
            psnr_sum += psnr_torch(o, c); ssim_sum += ssim_torch(o, c); cnt += 1
    psnr_avg = psnr_sum / max(1, cnt); ssim_avg = ssim_sum / max(1, cnt)
    print(f"[TEST] PSNR={psnr_avg:.2f}, SSIM={ssim_avg:.4f}")
    return psnr_avg, ssim_avg

def main():
    p = argparse.ArgumentParser()
    # 数据路径
    p.add_argument('--train_noisy', type=str, required=True)
    p.add_argument('--train_clean', type=str, required=True)
    p.add_argument('--test_noisy',  type=str, required=True)
    p.add_argument('--test_clean',  type=str, required=True)
    p.add_argument('--out_dir',     type=str, required=True)
    p.add_argument('--ckpt_dir',    type=str, required=True)
    # 模型与结构
    p.add_argument('--model', type=str, default='swin_only', choices=['swin_only','unet','resunet','stresunet'])
    p.add_argument('--width', type=int, default=96)
    p.add_argument('--swin_heads', type=str, default='8')       # swin-only 默认用一个 heads
    p.add_argument('--swin_depths', type=str, default='6')      # swin-only 默认深度6
    p.add_argument('--win', type=int, default=8)
    p.add_argument('--channels', type=int, default=1)
    # 训练控制
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--patience_hit', type=int, default=20)
    p.add_argument('--crop_size', type=int, default=256)        # 建议为 win 的倍数
    p.add_argument('--resize', type=int, default=0)
    p.add_argument('--loss', type=str, default='charb_ms', choices=['charb_ms','l1_ms','charb','l1'])
    p.add_argument('--ssim_w', type=float, default=0.1)
    # 其它
    p.add_argument('--residual', action='store_true')
    p.add_argument('--residual_scale', type=float, default=0.1)
    p.add_argument('--use_y_psnr', action='store_true')  # 1通道时等价普通PSNR
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--tta', action='store_true')
    args = p.parse_args()

    ensure_dir(Path(args.out_dir)); ensure_dir(Path(args.ckpt_dir))
    if args.eval_only:
        evaluate(args, model=None)
    else:
        model = train_one(args)
        evaluate(args, model=model)  # 即使不加 --tta 也跑一次标准测试，便于解析

if __name__ == '__main__':
    main()
