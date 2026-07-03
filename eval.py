import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader
from tqdm import tqdm

from semilearn.algorithms.dac.dac import OSCNet
from semilearn.algorithms.ggriomatch.ggriomatch import GGRIOMatchNet
from semilearn.algorithms.iomatch.iomatch import IOMatchNet
from semilearn.algorithms.mtc.mtc import MTCNet
from semilearn.algorithms.openmatch.openmatch import OpenMatchNet
from semilearn.algorithms.uagreg.uagreg import UAGreg_Net
from semilearn.core.utils import get_dataset, get_net_builder, over_write_args_from_file


def parse_args():
    parser = argparse.ArgumentParser(description="Open-set SSL evaluation")
    parser.add_argument("--c", type=str, required=True, help="config file path")
    parser.add_argument("--load_path", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--step", type=str, default="best", help="checkpoint tag, kept for compatibility")
    parser.add_argument("--use_ema", action="store_true", default=False, help="load ema_model instead of model")
    parser.add_argument("--extended_test", action="store_true", default=False, help="use extended open-set test split")
    parser.add_argument("--testset", type=str, default="test", choices=["test", "unlabeled"])
    parser.add_argument("--batch_size", type=int, default=None, help="override evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg_args = argparse.Namespace()
    over_write_args_from_file(cfg_args, args.c)
    for key, value in vars(args).items():
        setattr(cfg_args, key, value)

    cfg_args.correlated_ood = getattr(cfg_args, "correlated_ood", False)
    cfg_args.mm_ablation = getattr(cfg_args, "mm_ablation", False)
    cfg_args.ratio = getattr(cfg_args, "ratio", None)
    cfg_args.data_dir = getattr(cfg_args, "data_dir", "./data")
    cfg_args.net_from_name = getattr(cfg_args, "net_from_name", False)
    cfg_args.num_heads = getattr(cfg_args, "num_heads", 10)
    cfg_args.proj_dim = getattr(cfg_args, "proj_dim", 128)
    cfg_args.eval_batch_size = args.batch_size or getattr(cfg_args, "eval_batch_size", 256)
    return cfg_args


def load_model(args):
    checkpoint = torch.load(args.load_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["ema_model"] if args.use_ema and "ema_model" in checkpoint else checkpoint.get("model", checkpoint)

    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state[key.split(".", 1)[1]] = value
        else:
            cleaned_state[key] = value

    net_builder = get_net_builder(args.net, args.net_from_name)
    net = net_builder(num_classes=args.num_classes)

    if args.algorithm in {"openmatch", "ggropenmatch"}:
        net = OpenMatchNet(net, args.num_classes)
    elif args.algorithm in {"dac", "ggrdac"}:
        net = OSCNet(net, args.num_classes, num_heads=args.num_heads, proj_dim=args.proj_dim)
    elif args.algorithm == "ggriomatch":
        net = GGRIOMatchNet(
            net,
            args.num_classes,
            proj_size=getattr(args, "proj_size", 128),
            use_rot=getattr(args, "use_rot", False),
        )
    elif args.algorithm == "iomatch":
        net = IOMatchNet(net, args.num_classes)
    elif args.algorithm == "mtc":
        net = MTCNet(net)
    elif args.algorithm == "uagreg":
        net = UAGreg_Net(net, args.num_classes)

    net.load_state_dict(cleaned_state, strict=False)
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    return net


def calculate_ed(logits):
    return torch.logsumexp(logits, dim=1) * (1 - 1 / torch.exp(logits).max(dim=1)[0])


def calculate_consensus(probs_comm, temp_d=1.0, mode="l1"):
    _, num_heads, _ = probs_comm.shape
    if mode == "l1":
        consensus = (probs_comm.unsqueeze(1) - probs_comm.unsqueeze(2)).abs().mean([-3, -2, -1])
        return torch.exp(-consensus / temp_d)

    marginal_p = probs_comm.mean(dim=0)
    marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)
    marginal_p = rearrange(marginal_p, "h g d e -> 1 (h g) (d e)")
    pointwise_p = torch.einsum("bhd,bge->bhgde", probs_comm, probs_comm)
    pointwise_p = rearrange(pointwise_p, "b h g d e -> b (h g) (d e)")
    kl_computed = pointwise_p * (pointwise_p.log() - marginal_p.log())
    kl_grid = rearrange(kl_computed.sum(-1), "b (h g) -> b h g", h=num_heads)
    return torch.triu(kl_grid, diagonal=1).mean([-1, -2])


def get_loader(args, dataset_dict):
    if args.testset == "unlabeled":
        return DataLoader(
            dataset_dict["train_ulb"],
            batch_size=args.eval_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=args.num_workers,
        )

    test_key = "extended" if args.extended_test else "full"
    batch_size = 1024 if args.extended_test else args.eval_batch_size
    return DataLoader(
        dataset_dict["test"][test_key],
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
    )


def move_to_device(x):
    if isinstance(x, dict):
        return {k: v.cuda() for k, v in x.items()}
    return x.cuda()


def evaluate(args, net, dataset_dict):
    loader = get_loader(args, dataset_dict)
    labels_all = []
    preds_all = []
    scores_all = []
    aux_scores_all = []

    with torch.no_grad():
        for data in tqdm(loader):
            if args.testset == "unlabeled":
                x = data["x_ulb_w_0"] if args.algorithm in {"mtc", "openmatch"} else data["x_ulb_w"]
                y = data["y_ulb"]
            else:
                x = data["x_lb"]
                y = data["y_lb"]

            x = move_to_device(x) if torch.cuda.is_available() else x
            y = y.cuda() if torch.cuda.is_available() else y

            outputs = net(x)
            logits = outputs["logits"]
            closed_scores, closed_preds = torch.max(F.softmax(logits, dim=-1), dim=-1)

            if args.algorithm in {"openmatch", "ggropenmatch"}:
                logits_open = outputs["logits_open"]
                probs_open = F.softmax(logits_open.view(logits_open.size(0), 2, -1), 1)
                idx = torch.arange(logits_open.size(0), device=logits_open.device)
                score = probs_open[idx, 0, closed_preds]

            elif args.algorithm in {"dac", "ggrdac"}:
                logits_comm = outputs["logits_comm"].view(-1, args.num_heads, args.num_classes)
                probs_comm = torch.softmax(logits_comm, dim=-1)
                score = calculate_consensus(probs_comm)
                aux_scores_all.extend(probs_comm.mean(dim=1).max(dim=-1)[0].cpu().tolist())

            elif args.algorithm in {"iomatch", "ggriomatch"}:
                logits_mb = outputs["logits_mb"]
                closed_prob = F.softmax(logits, dim=1)
                mb_prob = F.softmax(logits_mb.view(logits_mb.size(0), 2, -1), 1)
                outlier_prob = mb_prob[:, 0, :]
                inlier_prob = mb_prob[:, 1, :]
                hat_q = torch.zeros((logits.size(0), args.num_classes + 1), device=logits.device)
                hat_q[:, :args.num_classes] = closed_prob * inlier_prob
                hat_q[:, args.num_classes] = torch.sum(closed_prob * outlier_prob, 1)
                score, _ = torch.max(hat_q[:, :args.num_classes], dim=1)

            elif args.algorithm == "mtc":
                score = torch.sigmoid(outputs["domain_logits"]).squeeze(-1)

            elif args.algorithm == "safestudent":
                score = calculate_ed(logits)

            elif args.algorithm == "uagreg":
                logits_open = outputs["logits_open"]
                score, _ = torch.max(torch.softmax(logits_open, dim=1), dim=-1)

            else:
                score = closed_scores

            labels_all.extend(y.cpu().tolist())
            preds_all.extend(closed_preds.cpu().tolist())
            scores_all.extend(score.cpu().tolist())

    labels = torch.tensor(labels_all).long()
    preds = torch.tensor(preds_all).long()
    scores = torch.tensor(scores_all).float()
    id_mask = labels.lt(args.num_classes)
    ood_mask = labels.ge(args.num_classes)
    labels[ood_mask] = args.num_classes

    threshold_scores = scores.clone()
    if args.algorithm in {"dac", "ggrdac"} and aux_scores_all:
        aux_scores = torch.tensor(aux_scores_all).float()
        threshold_scores = (threshold_scores - threshold_scores.min()) / (threshold_scores.max() - threshold_scores.min() + 1e-8)
        aux_scores = (aux_scores - aux_scores.min()) / (aux_scores.max() - aux_scores.min() + 1e-8)

    threshold = threshold_otsu(threshold_scores.cpu().numpy()) if len(threshold_scores) > 0 else 0.95
    if args.algorithm in {"openmatch", "ggropenmatch"}:
        open_mask = threshold_scores.ge(threshold)
    elif args.algorithm in {"dac", "ggrdac", "iomatch", "ggriomatch", "mtc", "safestudent", "uagreg"}:
        open_mask = threshold_scores.lt(threshold)
    else:
        open_mask = threshold_scores.lt(0.95)

    open_preds = preds.clone()
    open_preds[open_mask] = args.num_classes

    close_acc = accuracy_score(labels[id_mask].cpu().numpy(), preds[id_mask].cpu().numpy())
    open_acc = balanced_accuracy_score(labels.cpu().numpy(), open_preds.cpu().numpy())

    print("#############################################################")
    print(f"Method:              {args.algorithm}")
    print(f"Config:              {args.c}")
    print(f"Checkpoint:          {args.load_path}")
    print(f"Dataset:             {args.dataset}")
    print(f"Num_classes:         {args.num_classes}")
    print(f"Num_labels:          {args.num_labels}")
    print(f"Closed-set Accuracy: {close_acc * 100:.2f}")
    print(f"Open-set Accuracy:   {open_acc * 100:.2f}")
    print("#############################################################")


def main():
    args = parse_args()
    dataset_dict = get_dataset(
        args,
        args.algorithm,
        args.dataset,
        args.num_labels,
        args.num_classes,
        args.data_dir,
        eval_open=True,
    )
    net = load_model(args)
    evaluate(args, net, dataset_dict)


if __name__ == "__main__":
    main()
