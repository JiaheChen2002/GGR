import copy
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from semilearn.core.utils import get_data_loader, ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import (
    ce_loss, consistency_loss, SSL_Argument, str2bool,
    flatten_grads_dense, unflatten_like_dense,
    GradSubspace, rectify_pair, select_surgery_params, normalize_surgery_mode,
)

from .utils import ova_loss_func, em_loss_func, socr_loss_func


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class OpenMatchDataset(BasicDataset):
    def __init__(self, dset, name):
        self.data = copy.deepcopy(dset.data)
        self.targets = copy.deepcopy(dset.targets)
        super(OpenMatchDataset, self).__init__(alg='openmatch',
                                               data=self.data,
                                               targets=self.targets,
                                               num_classes=dset.num_classes,
                                               transform=dset.transform,
                                               strong_transform=dset.strong_transform)
        self.name = name
        self.data_index = None
        self.targets_index = None
        self.set_index()

    def set_index(self, indices=None):
        if indices is not None:
            self.data_index = self.data[indices]
            self.targets_index = self.targets[indices]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def __len__(self):
        return len(self.data_index)

    def __sample__(self, idx):
        if self.targets is None:
            target = None
        else:
            target = self.targets_index[idx]
        img = self.data_index[idx]

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(img, str):
            img = pil_loader(img)

        return img, target

    def __getitem__(self, idx):
        img, target = self.__sample__(idx)

        img_w = self.transform(img)
        if self.name == 'train_lb':
            return {'idx_lb': idx, 'x_lb': img_w, 'x_lb_w_0': img_w, 'x_lb_w_1': self.transform(img),
                    'y_lb': target}
        elif self.name == 'train_ulb':
            return {'idx_ulb': idx, 'x_ulb_w_0': img_w, 'x_ulb_w_1': self.transform(img), 'y_ulb': target}
        elif self.name == 'train_ulb_selected':
            return {'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)}


class OpenMatchNet(nn.Module):
    def __init__(self, base, num_classes):
        super(OpenMatchNet, self).__init__()
        self.backbone = base
        self.feat_planes = base.num_features
        self.ova_classifiers = nn.Linear(self.feat_planes, num_classes * 2, bias=False)

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        logits_open = self.ova_classifiers(feat)
        return {'logits': logits, 'logits_open': logits_open}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@ALGORITHMS.register('ggropenmatch')
class GGROpenMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.p_cutoff = args.p_cutoff
        self.lambda_em = args.lambda_em
        self.lambda_socr = args.lambda_socr
        self.start_fix = args.start_fix
        self.fix_uratio = args.fix_uratio

        self.use_surgery = args.use_surgery
        self.surgery_mode = normalize_surgery_mode(args.surgery_mode)
        self.surgery_scope = args.surgery_scope
        self.subspace_dim = args.subspace_dim
        self.subspace_update_interval = max(1, args.subspace_update_interval)
        self.surgery_alpha = args.surgery_alpha

        self.grad_subspace = None
        self._iter = 0
        self.cum_conflict_regret = 0.0

    def set_dataset(self):
        dataset_dict = super(GGROpenMatch, self).set_dataset()
        dataset_dict['train_lb'] = OpenMatchDataset(dset=dataset_dict['train_lb'], name='train_lb')
        dataset_dict['train_ulb'] = OpenMatchDataset(dset=dataset_dict['train_ulb'], name='train_ulb')
        dataset_dict['train_ulb_selected'] = OpenMatchDataset(dset=dataset_dict['train_ulb'], name='train_ulb_selected')
        return dataset_dict

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = OpenMatchNet(model, num_classes=self.num_classes)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = OpenMatchNet(ema_model, num_classes=self.num_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def _rectify_pair(self, g_sup_vec, g_aux_vec):
        return rectify_pair(
            g_sup_vec, g_aux_vec,
            mode=self.surgery_mode,
            grad_subspace=self.grad_subspace,
            subspace_dim=self.subspace_dim,
            surgery_alpha=self.surgery_alpha,
        )

    def train(self):
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch

            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            if self.epoch != 0:
                self.exclude_dataset()

            self.loader_dict['train_ulb_selected'] = get_data_loader(self.args,
                                                                     self.dataset_dict['train_ulb_selected'],
                                                                     self.args.batch_size * self.fix_uratio,
                                                                     data_sampler=self.args.train_sampler,
                                                                     num_iters=self.num_train_iter // self.epochs,
                                                                     num_epochs=1,
                                                                     num_workers=2 * self.args.num_workers,
                                                                     distributed=self.distributed)

            for data_lb, data_ulb, data_ulb_selected in zip(self.loader_dict['train_lb'],
                                                            self.loader_dict['train_ulb'],
                                                            self.loader_dict['train_ulb_selected']):
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.tb_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb, **data_ulb_selected))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, x_lb_w_0, x_lb_w_1, y_lb, x_ulb_w_0, x_ulb_w_1, x_ulb_w, x_ulb_s):
        self._iter += 1
        num_lb = y_lb.shape[0]
        device = y_lb.device

        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w_0, x_lb_w_1, x_ulb_w_0, x_ulb_w_1))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb * 2]
                logits_open_lb = outputs['logits_open'][:num_lb * 2]
                logits_open_ulb_0, logits_open_ulb_1 = outputs['logits_open'][num_lb * 2:].chunk(2)
            else:
                raise ValueError("Bad configuration: use_cat should be True!")

            sup_loss = ce_loss(logits_x_lb, y_lb.repeat(2), reduction='mean')
            ova_loss = ova_loss_func(logits_open_lb, y_lb.repeat(2))
            sup_loss_total = sup_loss + ova_loss

            em_loss = em_loss_func(logits_open_ulb_0, logits_open_ulb_1)
            socr_loss = socr_loss_func(logits_open_ulb_0, logits_open_ulb_1)

            fix_loss = torch.tensor(0.0).to(device)
            mask_ratio = 0.0

            if self.epoch >= self.start_fix:
                inputs_selected = torch.cat((x_ulb_w, x_ulb_s), 0)
                outputs_selected = self.model(inputs_selected)
                logits_x_ulb_w, logits_x_ulb_s = outputs_selected['logits'].chunk(2)
                probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)

                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
                mask_ratio = mask.float().mean().item()

                pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                              logits=probs_x_ulb_w,
                                              use_hard_label=True,
                                              softmax=False)

                fix_loss = consistency_loss(logits_x_ulb_s,
                                            pseudo_label,
                                            'ce',
                                            mask=mask)

            aux_loss_total = self.lambda_em * em_loss + self.lambda_socr * socr_loss + fix_loss
            total_loss = sup_loss_total + aux_loss_total

        is_warmup = (self.epoch < self.start_fix)
        if (not self.use_surgery) or (aux_loss_total.item() == 0.0) or is_warmup:
            self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)
            cos_final = 0.0
        else:
            all_params, surgery_mask = select_surgery_params(self.model, self.surgery_scope)

            g_sup_all = torch.autograd.grad(sup_loss_total, all_params, retain_graph=True, create_graph=False, allow_unused=True)
            g_aux_all = torch.autograd.grad(aux_loss_total, all_params, retain_graph=False, create_graph=False, allow_unused=True)

            params_sel = [p for p, m in zip(all_params, surgery_mask) if m]
            g_sup_sel = [g for g, m in zip(g_sup_all, surgery_mask) if m]
            g_aux_sel = [g for g, m in zip(g_aux_all, surgery_mask) if m]

            g_sup_vec, shapes = flatten_grads_dense(params_sel, g_sup_sel, device=device)
            g_aux_vec, _ = flatten_grads_dense(params_sel, g_aux_sel, device=device)

            if self.subspace_dim > 0 and self.grad_subspace is None:
                self.grad_subspace = GradSubspace(max_dim=self.subspace_dim, device=device)
            if self.grad_subspace is not None and (self._iter % self.subspace_update_interval == 0):
                self.grad_subspace.update(g_sup_vec)

            g_sup_safe_vec, g_aux_safe_vec = self._rectify_pair(g_sup_vec, g_aux_vec)
            g_sup_safe_sub = unflatten_like_dense(g_sup_safe_vec, shapes, device=device)
            g_aux_safe_sub = unflatten_like_dense(g_aux_safe_vec, shapes, device=device)
            safe_sup_it = iter(g_sup_safe_sub)
            safe_aux_it = iter(g_aux_safe_sub)

            cos_final = 0.0
            with torch.no_grad():
                if g_sup_safe_vec is not None and g_aux_safe_vec is not None:
                    dot = torch.dot(g_sup_safe_vec, g_aux_safe_vec)
                    n1 = g_sup_safe_vec.norm()
                    n2 = g_aux_safe_vec.norm()
                    if n1 > 0 and n2 > 0:
                        cos_final = float(dot / (n1 * n2))
                    if dot < 0:
                        self.cum_conflict_regret += float(-dot)

            g_final = []
            for gs, gu, m in zip(g_sup_all, g_aux_all, surgery_mask):
                if m:
                    gs_new = next(safe_sup_it)
                    gu_new = next(safe_aux_it)
                else:
                    gs_new = gs
                    gu_new = gu

                if gs_new is None:
                    g = gu_new
                elif gu_new is None:
                    g = gs_new
                else:
                    g = gs_new + gu_new
                g_final.append(g)

            surrogate = torch.zeros((), device=device, dtype=torch.float32)
            for p, g in zip(all_params, g_final):
                if g is not None:
                    surrogate = surrogate + torch.sum(p * g.detach())
            self.call_hook("param_update", "ParamUpdateHook", loss=surrogate)

        tb_dict = {'train/sup_loss': sup_loss.item(), 'train/ova_loss': ova_loss.item(),
                   'train/em_loss': em_loss.item(), 'train/socr_loss': socr_loss.item(),
                   'train/total_loss': total_loss.item(),
                   'analysis/cum_conflict_regret': self.cum_conflict_regret,
                   'analysis/grad_cos_final': float(cos_final)}

        if self.epoch >= self.start_fix:
            tb_dict['fix_loss'] = fix_loss.item()
            tb_dict['mask_ratio'] = mask_ratio

        return tb_dict

    def exclude_dataset(self):
        loader = DataLoader(dataset=self.dataset_dict['train_ulb'],
                            batch_size=self.args.eval_batch_size,
                            drop_last=False,
                            shuffle=False,
                            num_workers=4)

        self.model.eval()
        self.ema.apply_shadow()
        self.print_fn(f"Selecting...")
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                x = data['x_ulb_w_0']
                y = data['y_ulb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                outputs = self.model(x)
                logits, logits_open = outputs['logits'], outputs['logits_open']
                logits = F.softmax(logits, 1)
                logits_open = F.softmax(logits_open.view(logits_open.size(0), 2, -1), 1)
                tmp_range = torch.arange(0, logits_open.size(0)).long().cuda(self.gpu)
                pred_close = logits.data.max(1)[1]
                unk_score = logits_open[tmp_range, 0, pred_close]
                select_idx = unk_score < 0.5
                gt_idx = y < self.args.num_classes
                if batch_idx == 0:
                    select_all = select_idx
                    gt_all = gt_idx
                else:
                    select_all = torch.cat([select_all, select_idx], 0)
                    gt_all = torch.cat([gt_all, gt_idx], 0)

        select_accuracy = accuracy_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_precision = precision_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_recall = recall_score(gt_all.cpu().numpy(), select_all.cpu().numpy())

        select_all = select_all.cpu().numpy()
        selected_idx = torch.arange(0, len(select_all))[select_all]
        if self.rank == 0:
            self.print_fn(f"Selected ratio = {len(selected_idx) / len(select_all)}, accuracy = {select_accuracy}, "
                          f"precision = {select_precision}, recall = {select_recall}")

        self.ema.restore()
        self.model.train()
        if self.epoch >= self.start_fix:
            if len(selected_idx) > 0:
                self.dataset_dict['train_ulb_selected'].set_index(selected_idx)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--lambda_em', float, 0.1),
            SSL_Argument('--lambda_socr', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.0),
            SSL_Argument('--start_fix', int, 10),
            SSL_Argument('--fix_uratio', int, 7),

            SSL_Argument('--use_surgery', str2bool, True),
            SSL_Argument('--surgery_mode', str, 'vlr'),
            SSL_Argument('--surgery_scope', str, 'backbone'),
            SSL_Argument('--subspace_dim', int, 10),
            SSL_Argument('--subspace_update_interval', int, 1),
            SSL_Argument('--surgery_alpha', float, 1.0),
        ]
