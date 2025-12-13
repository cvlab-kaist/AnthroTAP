<div align="center">
<h2>AnthroTAP: Learning Point Tracking with Real-World Motion</h2>

[**Inès Hyeonsu Kim**](https://sites.google.com/view/ines-hyeonsu-kim/home)<sup>1,3*</sup> · [**Seokju Cho**](https://sites.google.com/view/seokjucho/home)<sup>1*</sup> · [**Jahyeok Koo**](https://scholar.google.com/citations?user=1Vl37dcAAAAJ&hl=ko)<sup>1</sup> · [**Junghyun Park**](https://junghyun-james-park.github.io/)<sup>1</sup><br>[**Jiahui Huang**](https://gabriel-huang.github.io)<sup>2</sup> · [**Honglak lee**](https://web.eecs.umich.edu/~honglak/)<sup>3,4</sup> · [**Joon-Young Lee**](https://joonyoung-cv.github.io)<sup>2</sup>  · [**Seungryong Kim**](https://cvlab.kaist.ac.kr/)<sup>1</sup>

<sup>1</sup>KAIST AI&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Adobe Research&emsp;&emsp;&emsp;&emsp;<sup>3</sup>University of Michigan&emsp;&emsp;&emsp;&emsp;<sup>4</sup>LG AI Research

<span style="font-size: 1.5em;"><b>arXiv 2025</b></span>

<a href="https://arxiv.org/abs/2507.06233"><img src='https://img.shields.io/badge/arXiv-AnthroTAP-red' alt='Paper PDF'></a>
<a href='https://cvlab-kaist.github.io/AnthroTAP/'><img src='https://img.shields.io/badge/Project_Page-AnthroTAP-green' alt='Project Page'></a>


</div>


## Preparing the Environment

```bash
git clone https://github.com/cvlab-kaist/AnthroTAP
cd AnthroTAP/locotrack_pytorch/
git clone https://github.com/google-research/kubric.git # Optional, only for training.

conda create -n locotrack-pytorch python=3.11
conda activate locotrack-pytorch

pip install torch torchvision torchaudio lightning==2.3.3 tensorflow_datasets tensorflow matplotlib mediapy tensorflow_graphics einops wandb
```

## LocoTrack Evaluation

### 1. Download Pre-trained Weights

To evaluate LocoTrack on the benchmarks, first download the pre-trained weights from [Link](https://drive.google.com/file/d/1Rj7sIby_ylZkuy4pccAA28dtqvJQUH-A).

Alternatively, you can download the pre-trained weights using the following command:

```bash
gdown 1Rj7sIby_ylZkuy4pccAA28dtqvJQUH-A
```

### 3. Run Evaluation

To evaluate the LocoTrack model, use the `experiment.py` script with the following command-line arguments:

```bash
python experiment.py --config config/default.ini --mode eval_{dataset_to_eval_1}_..._{dataset_to_eval_N}[_q_first] --ckpt_path /path/to/checkpoint --save_path ./path_to_save_checkpoints/
```

- `--config`: Specifies the path to the configuration file. Default is `config/default.ini`.
- `--mode`: Specifies the mode to run the script. Use `eval` to perform evaluation. You can also include additional options for query first mode (`q_first`), and the name of the evaluation datasets. For example:
  - Evaluation of the DAVIS dataset: `eval_davis`
  - Evaluation of DAVIS and RoboTAP in query first mode: `eval_davis_robotap_q_first`
- `--ckpt_path`: Specifies the path to the checkpoint file. If not provided, the script will use the default checkpoint.
- `--save_path`: Specifies the path to save logs. 

Replace `/path/to/checkpoint` with the actual path to your checkpoint file. This command will run the evaluation process and save the results in the specified `save_path`.

Example:
```bash
python experiment.py \
--config config/default.ini \
--mode eval_davis_q_first \
--save_path ./log \
--ckpt_path ./Anthro-LocoTrack.ckpt
```
