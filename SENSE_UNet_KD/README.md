# U-Net Knowledge Distillation

Separate training pipeline for crowd-counting distillation with:

- Lightweight U-Net student (`models/unet_student.py`)
- Pretrained teacher wrapper (`models/teacher_wrapper.py`)
- Distillation losses (`losses/distill_losses.py`)
- Dataset + augmentations (`data/ucf_cc50_dataset.py`)
- Train/validation loop (`train_unet_distill.py`)

## Run

```bash
python SENSE_UNet_KD/train_unet_distill.py --epochs 220 --batch_size 2 --use_split_lr
```

Optional SSIM:

```bash
python SENSE_UNet_KD/train_unet_distill.py --use_ssim --ssim_weight 0.05
```
