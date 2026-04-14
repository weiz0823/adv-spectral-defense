# Adversarial Spectral Defense (ASD)

[![arXiv](https://img.shields.io/badge/arXiv-2604.10715-b31b1b.svg)](https://arxiv.org/abs/2604.10715)
[![DOI](https://img.shields.io/badge/DOI-10.1109/TIFS.2026.3684298-blue.svg)](https://doi.org/10.1109/TIFS.2026.3684298)
[![repo](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/weiz0823/adv-spectral-defense)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Official code repository for [*Defending against Patch-Based and Texture-Based Adversarial Attacks with Spectral Decomposition*](https://arxiv.org/abs/2604.10715) (IEEE TIFS, 2026).

## Quickstart

The core of ASD is in `asd_defense/`. It is wrapped as an image preprocessing module `DWTPreprocessPlugin` (the ASD Masking part in our paper).
If you are using it, you can simply import it as follows:

```python
from asd_defense import DWTPreprocessPlugin
```

It has very simple dependencies on torch, torchvision, and `pywavelets` (we recommend `pywavelets>=1.7.0`).

Other parts of the repository make an adversarial person detection framework.

## Run ASD in the framework

The framework is built upon [weiz0823/adv-clothes-break-multiple-defenses](https://github.com/weiz0823/adv-clothes-break-multiple-defenses). Thus, we only provide instructions for some easy-to-run attacks, and please refer to that repository for more instructions.

### Evaluate on adversarial patches

Some adversarial patches are provided in `artifacts/`, including `adv_patch_frcnn.png` for patch against standardly trained Faster R-CNN, `adv_patch_frcnn_at.png` for patch against adversarial training (AT),
and `adv_patch_frcnn_asd_at.png` for adaptive patch against AT+ASD. The evaluation process is detailed below.

**(1) Prepare the Inria Person dataset:**
First of all, you need the Inria Person dataset, available from:

```bash
curl ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar -o inria.tar
```

Then place the extracted dataset to `data/InriaPerson/`.

**(2) Download model weights:**

> *TL; DR:* Download from [Google Drive](https://drive.google.com/uc?export=download&id=1FZk4vSdfgt3YtNCkP_h9bMKut3hlwYOj) or [THU Cloud Storage](https://cloud.tsinghua.edu.cn/f/5164d1297d5b4fff8bf7/?dl=1), place it to `data/faster_rcnn_r50_fpn_at.pt`.

For the target model, we originally used the AT checkpoints from [thu-ml/oddefense](https://github.com/thu-ml/oddefense). It is built upon MMDetection, which has very complicated and currently outdated dependencies. Thus, we managed to convert the checkpoints to the Faster R-CNN architecture in torchvision (download converted AT checkpoint from [Google Drive](https://drive.google.com/uc?export=download&id=1FZk4vSdfgt3YtNCkP_h9bMKut3hlwYOj) or [THU Cloud Storage](https://cloud.tsinghua.edu.cn/f/5164d1297d5b4fff8bf7/?dl=1)). Then place the downloaded weight to `data/faster_rcnn_r50_fpn_at.pt`.

**(3) Setup environment:**
This project uses [`uv`](https://github.com/astral-sh/uv) to manage dependencies. Setup the environment with:

```bash
# Setup environment. Add -v for verbose output.
uv sync --group dwt
# Activate virtual environment
source .venv/bin/activate
```

Now you are ready to run the evaluation!

**(4) Run the evaluation:**

```bash
python tools/main.py -c configs/adv_patch.yaml configs/model/frcnn_at.yaml configs/defense/asd.yaml \
  --eval --patch artifacts/adv_patch_frcnn_asd_at.png
```

You will get a result similar to that in the paper (with acceptable numerical difference).

To get more accurate results, please configure `mmdet<3.0.0`, and use the original checkpoint from [thu-ml/oddefense](https://github.com/thu-ml/oddefense), which our code also supports.

**(5) Evaluate the performance without ASD:**
As a comparison, you can evaluate the performance without ASD:

```bash
# AT
python tools/main.py -c configs/adv_patch.yaml configs/model/frcnn_at.yaml --eval --patch artifacts/adv_patch_frcnn_at.png
# Standardly trained
python tools/main.py -c configs/adv_patch.yaml --eval --patch artifacts/adv_patch_frcnn.png
```

Check the clean performance with:

```bash
python tools/main.py -c configs/adv_patch.yaml --eval_clean
```

### Run the attack generation

You may run the attack generation with the provided code. Run the following command:

```bash
python tools/main.py -c configs/adv_patch.yaml
```

The results will be saved in `output/`.

### Advanced use

**Other attacks:** The framework also supports other attacks, including:

- Patch-based attacks: routed to `tools/train_patch.py`;
- Texture-based attacks: routed to `tools/train_camou.py`;
- Noise-based attacks: routed to `tools/train_noise.py`;
- Simple demonstration on patch-based attack against classifier: routed to `tools/classifier_patch_attack.py`.

Please refer to the corresponding source code and also [weiz0823/adv-clothes-break-multiple-defenses](https://github.com/weiz0823/adv-clothes-break-multiple-defenses) for more details.

Four attack configurations are used by us, two patch-based, two texture-based:

- `adv_patch.yaml`: The adversarial patch attack.
- `adv_tshirt.yaml`: The adversarial t-shirt attack.
- `adv_texture.yaml`: The AdvTexture attack.
- `adv_cat.yaml`: The AdvCaT attack.

Once you have properly prepared the environment, these attacks are runnable.

**Other models:** The framework supports many other models, but they require different additional dependencies, including YOLOv9, MMDetection, transformers, etc. Please refer to the corresponding source code in `adv_person/models` for more details.

## `pre-commit` hooks

We use [`pre-commit`](https://github.com/pre-commit/pre-commit) to keep the codebase clean and consistent.

## Citation

If you find that our work is helpful to you, please star this project and consider citing:

```bibtex
@article{zhang2026adv-spectral-defense,
  title={Defending against Patch-Based and Texture-Based Adversarial Attacks with Spectral Decomposition},
  author={Zhang, Wei and Chang, Xinyu and Li, Xiao and Zhu, Yiming and Hu, Xiaolin},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026},
  publisher={IEEE}
}
```

## Contributors

Wei Zhang @weiz0823 and Xinyu Chang contributed to this codebase.

## Acknowledgements

- [adv-clothes-break-multiple-defenses](https://github.com/weiz0823/adv-clothes-break-multiple-defenses)
- [oddefense](https://github.com/thu-ml/oddefense)
- [AdvCaT](https://github.com/WhoTHU/Adversarial_camou)
- [NumbOD](https://github.com/CGCL-codes/NumbOD)
- [LGP](https://github.com/liguopeng0923/LGP)
