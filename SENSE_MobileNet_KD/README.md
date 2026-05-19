# SENSE -> MobileNetV3 Knowledge Distillation

This folder is a standalone student-training project created beside `OMAN`.

It distills from the pretrained `SENSE.pth` teacher (OMAN PET model) into a `MobileNetV3` crowd-density student using UCF-CC-50 at `dataset/UCF_CC_50`.

## Distillation objective

The total loss follows:

`L_total = alpha * L_count + beta * L_distill`

- `L_count`: student supervision from UCF ground truth (count + density).
- `L_distill`: student mimicry of teacher outputs (teacher pseudo-density + teacher intermediate feature map).

## Folder contents

- `train_distill.py`: training/validation entry point.
- `kd_dataset.py`: UCF-CC-50 loader and GT density map creation.
- `student_model.py`: MobileNetV3 student model and feature-distillation loss.

## Train

Run from this folder:

```bash
python train_distill.py --repo_root .. --teacher_ckpt ../OMAN/pretrained/SENSE.pth --ucf_root ../dataset/UCF_CC_50 --epochs 40
```

## Notes

- Teacher is frozen (`eval()` + `requires_grad=False`).
- Default batch size is `1` because the teacher inference path is configured for single-image processing.
- Best student checkpoint is saved at `checkpoints/student_best.pth`.
