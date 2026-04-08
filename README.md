# Infrared Self-Radiation Restoration

PyTorch implementation for infrared image denoising, deblurring, and self-radiation suppression.

## Files
- `make_dataset.py`: dataset generation script
- `train_denoise.py`: model training and testing script

## Environment
- Python 3.9
- PyTorch
- OpenCV
- NumPy
- torchvision

## Usage

Generate dataset:

```bash
python make_dataset.py --clean_dir clean --heatmap_dir xingretu --psf_dir PSF --output_dir dataset4000_new
```

Train and test:

```bash
python train_denoise.py --train_noisy path_to_train_noisy --train_clean path_to_train_clean --test_noisy path_to_test_noisy --test_clean path_to_test_clean --out_dir outputs --ckpt_dir checkpoints
```
