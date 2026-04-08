import os
import csv
import json
import random
import argparse
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAP_EXTS = IMG_EXTS | {".txt", ".csv", ".npy"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    if path is not None and path != "":
        os.makedirs(path, exist_ok=True)


def list_files(root, exts):
    files = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p))
    files.sort()
    return files


def norm01(x, eps=1e-8):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)


def clip01(x):
    return np.clip(x, 0.0, 1.0)


# =========================
# 中文路径 / 中文文件名 读写
# =========================
def cv_imread_gray_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def cv_imwrite_unicode(path, img):
    ensure_dir(os.path.dirname(path))
    ext = Path(path).suffix.lower()
    if ext == "":
        ext = ".png"
        path = str(path) + ".png"

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Cannot write image: {path}")

    buf.tofile(path)


def imread_gray_resize(path, size_hw):
    img = cv_imread_gray_unicode(path)
    h, w = size_hw
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


# =========================
# 文本矩阵读取
# =========================
def _read_text_auto_encoding(path):
    encodings = ["utf-8-sig", "gbk", "utf-8", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception as e:
            last_err = e
    raise ValueError(f"Cannot read text file: {path}, last error: {last_err}")


def _parse_numeric_text_matrix(path, min_numbers_per_line=8):
    """
    兼容：
    - 纯数字 txt/csv
    - 带表头的 txt/csv
    通过提取“数字足够多”的行，组成 2D 矩阵
    """
    text = _read_text_auto_encoding(path)
    num_pat = re.compile(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?')

    rows = []
    for line in text.splitlines():
        vals = num_pat.findall(line)
        if len(vals) >= min_numbers_per_line:
            try:
                row = [float(v) for v in vals]
                rows.append(row)
            except Exception:
                continue

    if len(rows) == 0:
        raise ValueError(f"No numeric matrix found in text file: {path}")

    row_lens = [len(r) for r in rows]
    common_len = Counter(row_lens).most_common(1)[0][0]
    rows = [r for r in rows if len(r) == common_len]

    if len(rows) == 0:
        raise ValueError(f"Failed to extract consistent numeric rows from: {path}")

    x = np.array(rows, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Parsed matrix is not 2D: {path}")

    return x


def _parse_zemax_psf_text(path):
    """
    兼容 Zemax FFT PSF 带表头文本
    返回：
        x: 2D array
        center_r, center_c: 中心点（0-based），若没解析到则为 None
    """
    text = _read_text_auto_encoding(path)

    center_r, center_c = None, None
    center_match = re.search(r"中心点是[:：]?\s*行\s*(\d+)\s*[，,]\s*列\s*(\d+)", text)
    if center_match:
        center_r = int(center_match.group(1)) - 1
        center_c = int(center_match.group(2)) - 1

    num_pat = re.compile(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?')
    rows = []
    for line in text.splitlines():
        vals = num_pat.findall(line)
        if len(vals) >= 8:
            try:
                row = [float(v) for v in vals]
                rows.append(row)
            except Exception:
                continue

    if len(rows) == 0:
        raise ValueError(f"No numeric PSF data found in text file: {path}")

    row_lens = [len(r) for r in rows]
    common_len = Counter(row_lens).most_common(1)[0][0]
    rows = [r for r in rows if len(r) == common_len]

    if len(rows) == 0:
        raise ValueError(f"Failed to extract consistent PSF rows from: {path}")

    x = np.array(rows, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Parsed PSF is not 2D: {path}")

    return x, center_r, center_c


# =========================
# 热图 / PSF 读取
# =========================
def load_map(path, size_hw):
    """
    读取热图 / 自辐射图：
    - 支持图片
    - 支持 npy
    - 支持 txt/csv
    - 做柔和压缩，避免热图盖住主体轮廓
    """
    ext = Path(path).suffix.lower()
    h, w = size_hw

    if ext in IMG_EXTS:
        x = imread_gray_resize(path, size_hw)

    elif ext == ".npy":
        x = np.load(path).astype(np.float32)
        if x.ndim != 2:
            raise ValueError(f"Map must be 2D: {path}")
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)

    else:
        x = _parse_numeric_text_matrix(path)
        if x.ndim != 2:
            raise ValueError(f"Map must be 2D: {path}")
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)

    x = norm01(x)

    # 压低中低亮度响应，只保留更强热区
    x = np.power(x, 2.2)

    # 再轻微平滑，让热图更像背景热雾，而不是大块硬纹理
    x = cv2.GaussianBlur(x, (0, 0), sigmaX=2.0, sigmaY=2.0)

    return clip01(x).astype(np.float32)


def crop_center_psf(x, center_r=None, center_c=None, crop_size=21):
    """
    从大 PSF 中裁出中心奇数核，避免偶数核中心偏移。
    推荐 21 或 31，不建议再用 65。
    """
    if crop_size is None:
        return x

    if crop_size <= 0:
        raise ValueError("crop_size must be > 0 or None")

    if crop_size % 2 == 0:
        raise ValueError("crop_size must be odd, e.g. 21, 31, 33")

    h, w = x.shape

    if h < crop_size or w < crop_size:
        return x

    if center_r is None:
        center_r = h // 2
    if center_c is None:
        center_c = w // 2

    half = crop_size // 2

    r1 = center_r - half
    r2 = center_r + half + 1
    c1 = center_c - half
    c2 = center_c + half + 1

    # 如果解析到的中心导致越界，则退回几何中心
    if r1 < 0 or c1 < 0 or r2 > h or c2 > w:
        center_r = h // 2
        center_c = w // 2
        r1 = center_r - half
        r2 = center_r + half + 1
        c1 = center_c - half
        c2 = center_c + half + 1

    return x[r1:r2, c1:c2]


def load_psf(path, crop_size=21):
    """
    读取 PSF：
    - 图片
    - npy
    - txt/csv（支持 Zemax FFT PSF 带表头）
    - 自动非负化 + 归一化
    - 默认裁成较小奇数核
    """
    ext = Path(path).suffix.lower()

    if ext in IMG_EXTS:
        x = cv_imread_gray_unicode(path).astype(np.float32)
        center_r, center_c = None, None

    elif ext == ".npy":
        x = np.load(path).astype(np.float32)
        center_r, center_c = None, None

    else:
        x, center_r, center_c = _parse_zemax_psf_text(path)

    if x.ndim != 2:
        raise ValueError(f"PSF must be 2D: {path}")

    x = np.maximum(x, 0.0)

    if crop_size is not None:
        x = crop_center_psf(x, center_r=center_r, center_c=center_c, crop_size=crop_size)

    s = float(x.sum())
    if s <= 1e-12:
        raise ValueError(f"Invalid PSF sum=0: {path}")

    return (x / s).astype(np.float32)


# =========================
# 增强与退化
# =========================
def augment_clean(img, rot_deg=8, trans_px=8, scale_min=0.97, scale_max=1.03, hflip_p=0.5):
    """
    对清晰图做较温和增强
    """
    h, w = img.shape

    did_hflip = (random.random() < hflip_p)
    if did_hflip:
        img = cv2.flip(img, 1)

    angle = random.uniform(-rot_deg, rot_deg)
    tx = random.uniform(-trans_px, trans_px)
    ty = random.uniform(-trans_px, trans_px)
    scale = random.uniform(scale_min, scale_max)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    out = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    meta = {
        "angle_deg": angle,
        "translate_x_px": tx,
        "translate_y_px": ty,
        "scale": scale,
        "hflip": 1 if did_hflip else 0
    }
    return out.astype(np.float32), meta


def sample_beta():
    """
    热图叠加强度进一步减小
    """
    return random.uniform(0.003, 0.015)


def add_gaussian(img, sigma):
    if sigma <= 0:
        return img
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return img + noise


def add_salt_pepper(img, prob):
    if prob <= 0:
        return img
    out = img.copy()
    rnd = np.random.rand(*img.shape)
    out[rnd < prob / 2.0] = 0.0
    out[rnd > 1.0 - prob / 2.0] = 1.0
    return out


def synthesize(clean_img, heatmap, psf=None):
    """
    目标：
    - 比现在更模糊
    - 但不要回到之前那种整片雾化、看不清轮廓
    - 热图仍然保持轻度
    """
    # 关键：把 blur_ratio 提高
    blur_ratio = random.uniform(0.4, 1.2)

    # 热图仍然保持轻
    beta = random.uniform(0.010, 0.10)

    # 很轻的高斯噪声
    sigma_g = random.uniform(0.0, 0.004)

    p_sp = 0.0

    if psf is None:
        blur = clean_img
    else:
        blur = cv2.filter2D(clean_img, -1, psf, borderType=cv2.BORDER_REFLECT101)

    # 主体模糊混合
    base = (1.0 - blur_ratio) * clean_img + blur_ratio * blur

    # 再加一个很轻的后处理模糊，让视觉上更接近“第二张略更模糊一点”
    post_sigma = random.uniform(0.6, 1.2)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=post_sigma, sigmaY=post_sigma)

    # 热图继续弱化
    heat_soft = np.clip(heatmap, 0.0, 1.0)
    heat_soft = np.power(heat_soft, 2.2)
    heat_soft = cv2.GaussianBlur(heat_soft, (0, 0), sigmaX=2.0, sigmaY=2.0)

    deg = base + beta * heat_soft
    deg = add_gaussian(deg, sigma_g)
    deg = clip01(deg)

    meta = {
        "blur_ratio": blur_ratio,
        "post_blur_sigma": post_sigma,
        "beta": beta,
        "gaussian_sigma_norm": sigma_g,
        "gaussian_sigma_255": sigma_g * 255.0,
        "salt_pepper_prob": p_sp
    }
    return deg.astype(np.float32), meta


# =========================
# 保存
# =========================
def save_img(path, img01):
    img_u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    cv_imwrite_unicode(path, img_u8)


def write_list(path, items):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(str(x) + "\n")


# =========================
# 主生成逻辑
# =========================
def generate_split(split_name, source_files, heatmap_files, psf_files, out_dir, size_hw, n_pairs, aug_cfg, psf_crop_size):
    split_root = os.path.join(out_dir, split_name)
    clean_dir = os.path.join(split_root, "clean")
    deg_dir = os.path.join(split_root, "degraded")
    ensure_dir(clean_dir)
    ensure_dir(deg_dir)

    rows = []

    for i in range(n_pairs):
        src_path = source_files[i % len(source_files)]
        heat_path = random.choice(heatmap_files)
        psf_path = random.choice(psf_files) if len(psf_files) > 0 else ""

        clean0 = imread_gray_resize(src_path, size_hw)
        clean_aug, aug_meta = augment_clean(
            clean0,
            rot_deg=aug_cfg["rot_deg"],
            trans_px=aug_cfg["trans_px"],
            scale_min=aug_cfg["scale_min"],
            scale_max=aug_cfg["scale_max"],
            hflip_p=aug_cfg["hflip_p"]
        )

        heat = load_map(heat_path, size_hw)
        psf = load_psf(psf_path, crop_size=psf_crop_size) if psf_path else None

        deg, deg_meta = synthesize(clean_aug, heat, psf)

        sid = f"{split_name}_{i:05d}"
        clean_save = os.path.join(clean_dir, f"{sid}_clean.png")
        deg_save = os.path.join(deg_dir, f"{sid}_degraded.png")

        save_img(clean_save, clean_aug)
        save_img(deg_save, deg)

        row = {
            "sample_id": sid,
            "split": split_name,
            "source_image": src_path,
            "heatmap": heat_path,
            "psf": psf_path,
            "clean_saved_path": clean_save,
            "degraded_saved_path": deg_save,
        }
        row.update(aug_meta)
        row.update(deg_meta)
        rows.append(row)

        if (i + 1) % 100 == 0 or (i + 1) == n_pairs:
            print(f"[{split_name}] {i + 1}/{n_pairs}")

    if len(rows) > 0:
        with open(os.path.join(split_root, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--clean_dir", type=str, required=True, help="清晰图目录")
    parser.add_argument("--heatmap_dir", type=str, required=True, help="热图/自辐射图目录")
    parser.add_argument("--psf_dir", type=str, default="", help="PSF目录，可为空")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="H W")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_pairs", type=int, default=3200)
    parser.add_argument("--val_pairs", type=int, default=400)
    parser.add_argument("--test_pairs", type=int, default=400)

    parser.add_argument("--train_source_count", type=int, default=10)
    parser.add_argument("--val_source_count", type=int, default=1)
    parser.add_argument("--test_source_count", type=int, default=2)

    parser.add_argument("--rot_deg", type=float, default=8.0)
    parser.add_argument("--trans_px", type=float, default=8.0)
    parser.add_argument("--scale_min", type=float, default=0.97)
    parser.add_argument("--scale_max", type=float, default=1.03)
    parser.add_argument("--hflip_p", type=float, default=0.5)

    parser.add_argument("--psf_crop_size", type=int, default=21, help="裁剪后的PSF核大小，建议 21 或 31，设为0表示不裁剪")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    clean_files = list_files(args.clean_dir, IMG_EXTS)
    heatmap_files = list_files(args.heatmap_dir, MAP_EXTS)
    psf_files = list_files(args.psf_dir, MAP_EXTS) if args.psf_dir else []

    if len(clean_files) == 0:
        raise ValueError("No clean images found.")
    if len(heatmap_files) == 0:
        raise ValueError("No heatmaps found.")

    if args.train_source_count + args.val_source_count + args.test_source_count > len(clean_files):
        raise ValueError("Source split exceeds number of clean images.")

    psf_crop_size = None if args.psf_crop_size == 0 else args.psf_crop_size
    if psf_crop_size is not None and psf_crop_size % 2 == 0:
        raise ValueError("--psf_crop_size must be odd, e.g. 21, 31, 33")

    rng = random.Random(args.seed)
    shuffled = clean_files[:]
    rng.shuffle(shuffled)

    train_sources = shuffled[:args.train_source_count]
    val_sources = shuffled[args.train_source_count: args.train_source_count + args.val_source_count]
    test_sources = shuffled[
        args.train_source_count + args.val_source_count:
        args.train_source_count + args.val_source_count + args.test_source_count
    ]

    write_list(os.path.join(args.output_dir, "splits", "train_sources.txt"), train_sources)
    write_list(os.path.join(args.output_dir, "splits", "val_sources.txt"), val_sources)
    write_list(os.path.join(args.output_dir, "splits", "test_sources.txt"), test_sources)

    aug_cfg = {
        "rot_deg": args.rot_deg,
        "trans_px": args.trans_px,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "hflip_p": args.hflip_p
    }

    size_hw = (args.image_size[0], args.image_size[1])

    generate_split(
        "train",
        train_sources,
        heatmap_files,
        psf_files,
        args.output_dir,
        size_hw,
        args.train_pairs,
        aug_cfg,
        psf_crop_size
    )

    generate_split(
        "val",
        val_sources,
        heatmap_files,
        psf_files,
        args.output_dir,
        size_hw,
        args.val_pairs,
        aug_cfg,
        psf_crop_size
    )

    generate_split(
        "test",
        test_sources,
        heatmap_files,
        psf_files,
        args.output_dir,
        size_hw,
        args.test_pairs,
        aug_cfg,
        psf_crop_size
    )

    summary = {
        "total_clean_source_images_found": len(clean_files),
        "total_heatmaps_found": len(heatmap_files),
        "total_psfs_found": len(psf_files),
        "train_source_count": len(train_sources),
        "val_source_count": len(val_sources),
        "test_source_count": len(test_sources),
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "test_pairs": args.test_pairs,
        "image_size": [args.image_size[0], args.image_size[1]],
        "augmentation": aug_cfg,
        "psf_crop_size": psf_crop_size,
        "degradation": {
            "blur_ratio_range": [0.10, 0.25],
            "beta_range": [0.003, 0.015],
            "heatmap_gamma": 2.2,
            "gaussian_sigma_norm_range": [0.0, 0.005],
            "salt_pepper_prob_range": [0.0, 0.0]
        }
    }

    with open(os.path.join(args.output_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()