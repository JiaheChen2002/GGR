# GGR: Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning

Official implementation of the ECCV 2026 paper
**"Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning"**.

GGR is a plug-in optimization framework that reformulates open-set SSL as
gradient-level control rather than sample-level selection. Given a supervised
gradient $g_s$ and an auxiliary gradient $g_u$, GGR rectifies $g_u$ so that
the applied auxiliary update is first-order non-opposing with respect to the
supervised anchor. Three rectifiers are provided:

| `surgery_mode` | Rectifier | Update rule |
|---|---|---|
| `vlr` (default) | Vector-Level Rectifier | $\tilde g_u = g_u - \dfrac{\langle g_u, g_s\rangle}{\lVert g_s \rVert^2} g_s$ when $\langle g_u, g_s\rangle < 0$, else $\tilde g_u = g_u$ |
| `osr` | Orthogonal Subspace Rectifier | $\tilde g_u = (I - UU^\top) g_u$ |
| `csr` | Conic Subspace Rectifier | $\tilde g_u = g_u - U\,\min(U^\top g_u, 0)$ |



## Installation

```bash
conda create -n ggr python=3.10 -y
conda activate ggr
pip install -r requirements.txt
```

## Dataset Layout

All datasets are expected under `./data`:

```text
./data
├── cifar10/cifar-10-batches-py
├── cifar100/cifar-100-python
├── cifar10_openset
├── cifar100_openset
├── imagenet30
└── ood_data
```


## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/ggriomatch/ggriomatch_cifar10_6_150_0.yaml
```

Helper scripts:

```bash
bash scripts/train_cifar10.sh 0 ggriomatch
bash scripts/train_cifar100_20.sh 0 ggropenmatch
bash scripts/train_cifar100_50.sh 0 ggrdac
bash scripts/train_in30.sh 0 ggriomatch
```

Checkpoints go to `./saved_models/openset_cv/<algorithm>/<save_name>/`.

### Config naming

Configs live under `config/openset_cv/<algorithm>/`.

- **CIFAR-10 / CIFAR-100 base**: `<algorithm>_<dataset>_<num_classes>_<num_labels>_<seed>.yaml`
- **ImageNet-30 base**: `<algorithm>_in30_<percent>_<seed>.yaml` with `percent` in `{p1, p5}`


## Evaluation

Closed-set accuracy on the in-distribution validation split is reported in
the training log at every `num_eval_iter` iterations, and the best checkpoint
is saved as `model_best.pth`. For the open-set evaluation, which measures
balanced accuracy across the in-distribution classes and the "unknown"
category on the full test set (optionally with auxiliary OOD data), run
`eval.py` against a saved checkpoint:

```bash
python eval.py \
  --c config/openset_cv/ggriomatch/ggriomatch_cifar10_6_150_0.yaml \
  --load_path saved_models/openset_cv/ggriomatch/ggriomatch_cifar10_6_150_0/model_best.pth
```

## Acknowledgments

Built on top of:
- [USB](https://github.com/microsoft/Semi-supervised-learning)
- [unified_ossl_benchmarks](https://github.com/heejokong/unified_ossl_benchmarks)
- [MTCF](https://github.com/YU1ut/Multi-Task-Curriculum-Framework-for-Open-Set-SSL)
- [OpenMatch](https://github.com/VisionLearningGroup/OP_Match)
- [Safe-Student](https://github.com/RUC-DWBI-ML/CVPR2022-SafeStudent)
- [IOMatch](https://github.com/nukezil/IOMatch)
- [UAGreg](https://github.com/heejokong/UAGreg)
- [DivCon (DAC)](https://github.com/heejokong/DivCon)

## Citation

```bibtex
@misc{chen2026geometricgradientrectificationsafe,
      title={Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning}, 
      author={Jiahe Chen and Qian Shao and Qiyuan Chen and Jiaying He and Jintai Chen and Jian Wu and Hongxia Xu},
      year={2026},
      eprint={2606.26973},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.26973}, 
}
```
