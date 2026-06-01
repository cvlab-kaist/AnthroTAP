<div align="center">
<h2>AnthroTAP: Learning Point Tracking with Real-World Motion</h2>

[**Inès Hyeonsu Kim**](https://sites.google.com/view/ines-hyeonsu-kim/home)<sup>1,3*</sup> · [**Seokju Cho**](https://sites.google.com/view/seokjucho/home)<sup>1*</sup> · [**Jahyeok Koo**](https://scholar.google.com/citations?user=1Vl37dcAAAAJ&hl=ko)<sup>1</sup> · [**Junghyun Park**](https://junghyun-james-park.github.io/)<sup>1</sup><br>[**Jiahui Huang**](https://gabriel-huang.github.io)<sup>2</sup> · [**Honglak Lee**](https://web.eecs.umich.edu/~honglak/)<sup>3,4</sup> · [**Joon-Young Lee**](https://joonyoung-cv.github.io)<sup>2</sup> · [**Seungryong Kim**](https://cvlab.kaist.ac.kr/)<sup>1</sup>

<sup>1</sup>KAIST AI&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Adobe Research&emsp;&emsp;&emsp;&emsp;<sup>3</sup>University of Michigan&emsp;&emsp;&emsp;&emsp;<sup>4</sup>LG AI Research

<span style="font-size: 1.5em;"><b>CVPR 2026</b></span>

<a href="https://arxiv.org/abs/2507.06233"><img src='https://img.shields.io/badge/arXiv-2507.06233-red' alt='Paper PDF'></a>
<a href='https://cvlab-kaist.github.io/AnthroTAP/'><img src='https://img.shields.io/badge/Project_Page-AnthroTAP-green' alt='Project Page'></a>

</div>

## Overview

**AnthroTAP** improves point tracking in human-centric scenarios by fine-tuning [LocoTrack](https://github.com/KU-CVLAB/locotrack) (ECCV 2024) on a curated dataset of real-world human motion videos with dense point annotations. Existing point trackers trained purely on synthetic data (e.g., Kubric) struggle with the complex, articulated motions of humans. AnthroTAP bridges this gap by introducing a real-world human motion dataset and a robust training strategy for point tracking.

## Preparing the Environment

```bash
git clone https://github.com/cvlab-kaist/AnthroTAP
cd AnthroTAP/locotrack_pytorch/

conda create -n anthrotap python=3.11
conda activate anthrotap

pip install torch torchvision torchaudio lightning==2.3.3 tensorflow_datasets tensorflow matplotlib mediapy tensorflow_graphics einops wandb decord scikit-image
```

> **Optional (for Kubric training data generation):**
> ```bash
> git clone https://github.com/google-research/kubric.git
> ```

## Evaluation

### 1. Download Pre-trained Weights

Download the AnthroTAP fine-tuned checkpoint:

| Model | Pre-trained Weights |
|-------|---------------------|
| Anthro-LocoTrack-B | [Link](https://drive.google.com/file/d/1Rj7sIby_ylZkuy4pccAA28dtqvJQUH-A) |
| Anthro-TAPNext | [Link](TODO) |

```bash
# Download Anthro-LocoTrack-B with gdown
gdown 1Rj7sIby_ylZkuy4pccAA28dtqvJQUH-A
```

### 2. Prepare Evaluation Datasets

```bash
# TAP-Vid-DAVIS
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip tapvid_davis.zip

# TAP-Vid-RGB-Stacking
wget https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip
unzip tapvid_rgb_stacking.zip

# RoboTAP
wget https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip
unzip robotap.zip
```

For TAP-Vid-Kinetics, follow the [TAP-Vid repository](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid).

### 3. Run Evaluation

```bash
cd locotrack_pytorch

python experiment.py \
  --config config/default.ini \
  --mode eval_davis_q_first \
  --save_path ./log \
  --ckpt_path ./Anthro-LocoTrack.ckpt
```

**`--mode` options:**
- `eval_davis` / `eval_davis_q_first`
- `eval_kinetics` / `eval_kinetics_q_first`
- `eval_robotics` / `eval_robotics_q_first`
- `eval_robotap` / `eval_robotap_q_first`
- Combine datasets: `eval_davis_robotap_q_first`

Set dataset paths in `config/default.ini` under `[TRAINING]-val_dataset_path`.

## Training

AnthroTAP jointly trains on synthetic Kubric data and real-world human motion data.

### 1. Prepare Synthetic Training Data (Kubric)

Download the panning-MOVi-E dataset (~273 GB) from HuggingFace (requires [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)):

```bash
git clone git@hf.co:datasets/hamacojr/LocoTrack-panning-MOVi-E
```

### 2. Prepare AnthroTAP Human Motion Data

Please download data from [Link](https://huggingface.co/datasets/hamacojr/AnthroTAP-LetsDance)

### 3. Configure Paths

Edit `locotrack_pytorch/config/default.ini`:

```ini
[TRAINING]
val_dataset_path = {"davis": "/path/to/tapvid_davis.pkl", ...}
kubric_dir      = /path/to/panning-MOVi-E
human_data_dir  = /path/to/anthrotap
```

### 4. Run Training

```bash
cd locotrack_pytorch

python experiment.py \
  --config config/default.ini \
  --mode train_davis \
  --save_path ./checkpoints \
  --ckpt_path ./Anthro-LocoTrack.ckpt  # optional: resume or fine-tune
```

For multi-GPU training and mixed precision, set `precision = bf16-mixed` in the config.

## Demo

Run the interactive Gradio demo locally:

```bash
pip install -r demo/requirements.txt
python demo/demo.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2026anthrotap,
  title     = {AnthroTAP: Learning Point Tracking with Real-World Motion},
  author    = {Kim, In{\`e}s Hyeonsu and Cho, Seokju and Koo, Jahyeok and Park, Junghyun and Huang, Jiahui and Lee, Honglak and Lee, Joon-Young and Kim, Seungryong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

We also encourage citing the base LocoTrack model:

```bibtex
@article{cho2024local,
  title   = {Local All-Pair Correspondence for Point Tracking},
  author  = {Cho, Seokju and Huang, Jiahui and Nam, Jisu and An, Honggyu and Kim, Seungryong and Lee, Joon-Young},
  journal = {arXiv preprint arXiv:2407.15420},
  year    = {2024}
}
```

## Acknowledgement

This project builds upon [LocoTrack](https://github.com/KU-CVLAB/locotrack) and the [TAP-Net repository](https://github.com/google-deepmind/tapnet). We thank the authors for their invaluable contributions.
