# Video Individual Counting With Implicit One-to-many Matching (ICIP 2025)

This repository includes the official implementation of the paper:

[**Video Individual Counting With Implicit One-to-many Matching**](https://arxiv.org/abs/2506.13067)

International Conference on Image Processing (ICIP), 2025

Xuhui Zhu<sup>1</sup>, Jing Xu<sup>2</sup>,Bingjie Wang<sup>3</sup>,Huikang Dai<sup>2</sup>,[Hao Lu](https://sites.google.com/site/poppinace/)<sup>1</sup>

<sup>1</sup>Huazhong University of Science and Technology, China

<sup>2</sup>FiberHome Telecommunication Technologies Co., Ltd., China

<sup>3</sup>University of Rochester, Rochester, USA

[[Paper]](https://arxiv.org/abs/2506.13067) | [[Code]](https://github.com/tiny-smart/OMAN)

![OMAN](pics/Pipeline.png)

## Overview

Video Individual Counting (VIC) aims to estimate pedestrian flux from a video. Existing VIC approaches, however, mainly follow a one-to-one (O2O) matching strategy where the same pedestrian must be exactly matched between frames, leading to sensitivity to appearance variations or missing detections. In this work, we show that the O2O matching could be relaxed to a one-to-many (O2M) matching problem, which better fits the problem nature of VIC and can leverage the social grouping behavior of walking pedestrians. We therefore introduce OMAN, a simple but effective VIC model with implicit One-to-Many mAtchiNg, featuring an implicit context generator and a one-to-many pairwise matcher. Experiments on the SenseCrowd and CroHD benchmarks show that OMAN achieves the state-of-the-art performance.

## Installation

Clone and set up the CGNet repository:

```
git clone https://github.com/tiny-smart/OMAN
cd OMAN
conda create -n OMAN python=3.9
conda activate OMAN
pip install -r requirements.txt
```


## Data Preparation

- SenseCrowd: Download the dataset from [Baidu disk](https://pan.baidu.com/s/1OYBSPxgwvRMrr6UTStq7ZQ?pwd=64xm#list/path=%2F) or from the original dataset [link](https://github.com/HopLee6/VSCrowd-Dataset).


## Inferrence

- Download ImageNet pretrained ConvNext[[baidu dist]](https://pan.baidu.com/s/1oxxcD6h-JiRdJ4VItHJIUQ?pwd=ubqt)[[Google drive]](https://drive.google.com/file/d/1tDGb3DAEITajJ5xlzYSxCa5x4dnTbfJ-/view?usp=sharing), and put it in ```pretrained``` folder. Or you can define your pre-trained model path in [models/backbones/backbone.py](models/backbones/backbone.py)
- To test OMAN on SenseCrowd dataset, run

```
python test.py
```


## Evaluation

- To evaluate the results after testing, run

```
python eval_metrics.py
```

## Simple GUI (Count + Flow Rate)

After inference creates `outputs/json/video_results_test.json`, you can inspect per-video inflow and flow rate in a desktop GUI:

```
python simple_vic_gui.py
```

In the GUI:
- Open a results JSON (default path is auto-loaded if present).
- Select a video in the left panel.
- Set FPS and sampling interval (default interval is 15 frames as in `test.py`).
- View per-step inflow, cumulative count, and instantaneous flow rate (persons/min).

## Web UI (Upload Video + Run Pretrained Model)

This project now includes a Streamlit web UI that runs inference directly from an uploaded video (without using precomputed JSON).

### 1) Prepare weights

Put these files in `pretrained/`:
- `pretrained/SENSE.pth`
- `pretrained/convnext_small_384_in22ft1k.pth`

### 2) Install Streamlit

If you have already installed requirements, install Streamlit once:

```
pip install streamlit==1.40.1
```

### 3) Launch Web UI

```
streamlit run web_ui_streamlit.py
```

Then in browser:
- upload a crowd video file,
- click **Run Inference**,
- see final count, per-step inflow/outflow, and flow rate chart.

### CPU speed tuning (important)

For CPU-only laptops, enable **Fast mode** in UI. Internally it uses:
- larger `sample_interval` (process fewer frames),
- frame downscaling (`resize_width`),
- `max_sampled_frames` cap.

Suggested starting point to target ~5-7 min runtime:
- `sample_interval`: 45
- `resize_width`: 512
- `max_sampled_frames`: 40

### Notes

- CPU-only machines are supported through a compatibility fallback in `video_inference.py` (it patches hardcoded `.cuda()` calls to CPU). Inference on CPU will be significantly slower than GPU.
- You can also run direct CLI inference for one video:

```
python video_inference.py --video "path/to/video.mp4" --sample_interval 15 --gpu 0
```


## Pretrained Models

- Environment:

```
python==3.9
pytorch==2.0.1
torchvision==0.15.2
```

- Models:

| Dataset | Model Link | MAE | MSE | WRAE |
| :-- | :-- | :-- | :-- | :-- |
| SenseCrowd | SENSE.pth[[Baidu disk]](https://pan.baidu.com/s/1ZWxReVf9QeePRTsVwu9sIg?pwd=9a8c)[[Google drive]](https://drive.google.com/file/d/1XKBnOscinhDot4blvQQtFx3IgrZQ8Wae/view?usp=sharing) | 8.58 | 16.80 | 10.89% |

## Citation

If you find this work helpful for your research, please consider citing:

```
@INPROCEEDINGS{11084398,
  author={Zhu, Xuhui and Xu, Jing and Wang, Bingjie and Dai, Huikang and Lu, Hao},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)}, 
  title={Video Individual Counting with Implicit One-to-Many Matching}, 
  year={2025},
  volume={},
  number={},
  pages={61-66},
  keywords={Legged locomotion;Pedestrians;Sensitivity;Codes;Image processing;Semantics;Benchmark testing;Generators;Standards;Context modeling;Video individual counting;pedestrian flux;semantic correspondence;one-to-many matching},
  doi={10.1109/ICIP55913.2025.11084398}}
```


## Permission

This code is for academic purposes only. Contact: Xuhui Zhu (XuhuiZhu@hust.edu.cn)

## Acknowledgement

We thank the authors of [CGNet](https://github.com/streamer-AP/CGNet) and [PET](https://github.com/cxliu0/PET) for open-sourcing their work.

