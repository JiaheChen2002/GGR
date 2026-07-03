import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import (
    ce_loss, consistency_loss, SSL_Argument, str2bool, concat_all_gather,
    flatten_grads_dense, unflatten_like_dense,
    GradSubspace, rectify_pair, select_surgery_params, normalize_surgery_mode,
)

from .utils import SoftWeightingHook, DiverseLoss
from .dac import OSCNet, OSCDataset


@ALGORITHMS.register('ggrdac')
class GGRDAC(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

        self.div_criterion = DiverseLoss()

        self.use_surgery = args.use_surgery
        self.surgery_mode = normalize_surgery_mode(args.surgery_mode)
        self.surgery_scope = args.surgery_scope
        self.subspace_dim = args.subspace_dim
        self.subspace_update_interval = max(1, args.subspace_update_interval)
        self.surgery_alpha = args.surgery_alpha

        self.grad_subspace = None
        self._iter = 0

    def set_dataset(self):
        dataset_dict = super().set_dataset()
        dataset_dict['train_lb'] = OSCDataset(dset=dataset_dict['train_lb'], name='train_lb')
        dataset_dict['train_ulb'] = OSCDataset(dset=dataset_dict['train_ulb'], name='train_ulb')
        return dataset_dict

    def set_model(self):
        model = super().set_model()
        model = OSCNet(model, num_classes=self.num_classes, num_heads=self.args.num_heads,
                       proj_dim=self.args.proj_dim, use_rot=self.args.use_rot)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = OSCNet(ema_model, num_classes=self.num_classes, num_heads=self.args.num_heads,
                           proj_dim=self.args.proj_dim, use_rot=self.args.use_rot)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_hooks(self):
        self.register_hook(DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'), "DistAlignHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(SoftWeightingHook(num_data=len(self.dataset_dict['train_ulb']), ema_alpha=self.args.ema_alpha, temp_d=self.args.temp_d,
                                             use_joint=self.args.use_joint, device=self.gpu, temp_w=self.args.temp_w), "WeightingHook")

        self.queue_size = int(self.args.K * (self.args.uratio) * self.args.batch_size) if self.args.dataset != 'imagenet' else self.args.K
        self.u_feats_bank = torch.randn(self.queue_size, self.args.proj_dim).to(self.gpu)
        self.u_probs_bank = torch.zeros(self.queue_size, self.num_classes + 1).to(self.gpu) / (self.num_classes + 1)
        self.ptr = torch.zeros(1, dtype=torch.long).to(self.gpu)
        self.u_feats_bank = F.normalize(self.u_feats_bank, dim=1)

        super().set_hooks()

    def _update_bank(self, feats, probs):
        feats = concat_all_gather(feats)
        probs = concat_all_gather(probs)
        batch_size = feats.size(0)
        ptr = int(self.ptr[0])
        if self.queue_size % batch_size == 0:
            self.u_feats_bank[ptr:ptr + batch_size] = feats
            self.u_probs_bank[ptr:ptr + batch_size] = probs
            self.ptr[0] = (ptr + batch_size) % self.queue_size
        else:
            end = min(ptr + batch_size, self.queue_size)
            self.u_feats_bank[ptr:end] = feats[:end - ptr]
            self.u_probs_bank[ptr:end] = probs[:end - ptr]
            self.ptr[0] = (ptr + batch_size) % self.queue_size

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

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
                if self.it >= self.num_train_iter:
                    break
                self.call_hook("before_train_step")
                self.tb_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1
            self.call_hook("after_train_epoch")
        self.call_hook("after_run")

    def train_step(self, x_lb_w_0, x_lb_w_1, y_lb, x_ulb_w, x_ulb_s, idx_ulb):
        self._iter += 1
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        device = y_lb.device

        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w_0, x_lb_w_1, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb * 2]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb * 2:].chunk(2)
                logits_comm = outputs['logits_comm'].view(-1, self.args.num_heads, self.args.num_classes)
                logits_comm_x_lb = logits_comm[:num_lb * 2]
                logits_comm_x_ulb_w, logits_comm_x_ulb_s = logits_comm[num_lb * 2:].chunk(2)
                feat_proj_x_ulb_w, feat_proj_x_ulb_s = outputs['feat_proj'][num_lb * 2:].chunk(2)
            else:
                raise ValueError("Bad configuration: use_cat should be True!")

            loss_s_main = ce_loss(logits_x_lb, y_lb.repeat(2), reduction='mean')
            loss_unk_s = 0.
            for hidx in range(self.args.num_heads):
                loss_unk_s += F.cross_entropy(logits_comm_x_lb[:, hidx, :], y_lb.repeat(2), reduction='mean')
            loss_unk_s /= self.args.num_heads

            sup_loss_total = loss_s_main + loss_unk_s

            weight = self.call_hook("weighting", "WeightingHook", logits_comm=logits_comm_x_ulb_w.detach(), idx=idx_ulb)

            with torch.no_grad():
                targets_p = F.softmax(logits_x_ulb_w, dim=-1).detach()
                targets_comm = F.softmax(logits_comm_x_ulb_w.view(-1, self.args.num_classes), dim=-1).detach()
                if self.args.dist_align:
                    targets_p = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=targets_p)
                    targets_comm = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=targets_comm)
                targets_open = torch.einsum('bc, b -> bc', targets_p, weight)
                targets_open = torch.cat([targets_open, (1. - weight).abs().unsqueeze(-1)], dim=1)

            loss_mi = self.div_criterion(logits_comm_x_ulb_w)

            mask_comm = self.call_hook("masking", "MaskingHook", cutoff=self.args.c_cutoff, logits_x_ulb=targets_comm, softmax_x_ulb=False)
            yh_comm_ulb = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=targets_comm, use_hard_label=True, softmax=False)
            loss_unk_reg = consistency_loss(logits_comm_x_ulb_s.view(-1, self.args.num_classes), yh_comm_ulb, 'ce', mask=mask_comm)

            if self.epoch >= self.args.start_fix:
                mask_id, weight_mask = self.call_hook("masking", "WeightingHook", idx=idx_ulb)
                probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
                mask_p = self.call_hook("masking", "MaskingHook", cutoff=self.args.p_cutoff, logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
                yh_ulb = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_w, use_hard_label=True, softmax=False)

                if self.args.mask_type == 'hard':
                    loss_u = consistency_loss(logits_x_ulb_s, yh_ulb, 'ce', mask=mask_p * mask_id)
                elif self.args.mask_type == 'soft':
                    loss_u = consistency_loss(logits_x_ulb_s, yh_ulb, 'ce', mask=mask_p * weight_mask)

                u_feats_bank = self.u_feats_bank.clone().detach()
                u_probs_bank = self.u_probs_bank.clone().detach()
                relation_qu = F.softmax(feat_proj_x_ulb_s @ u_feats_bank.T / self.args.temp_s, dim=-1)
                nn_qu = relation_qu @ u_probs_bank
                loss_kd = torch.sum(-nn_qu.log() * targets_open.detach(), dim=1).mean()
            else:
                mask_p = torch.zeros(num_ulb).to(device)
                loss_u = torch.tensor(0.0).to(device)
                loss_kd = torch.tensor(0.0).to(device)

            aux_loss_total = (self.args.lambda_u * loss_u) + \
                             (self.args.lambda_reg * loss_unk_reg) + \
                             (self.args.lambda_mi * loss_mi) + \
                             (self.args.lambda_kd * loss_kd)

            total_loss = sup_loss_total + aux_loss_total

        self._update_bank(feat_proj_x_ulb_w, targets_open)

        if (not self.use_surgery) or (aux_loss_total.item() == 0.0):
            self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)
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

        tb_dict = {
            'train/sup_loss': loss_s_main.item(),
            'train/unsup_loss': loss_u.item(),
            'train/total_loss': total_loss.item(),
            'train/weighted_mask': (mask_p * weight_mask).float().mean().item() if self.epoch >= self.args.start_fix else 0.0,
        }
        return tb_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--dist_align', str2bool, False),
            SSL_Argument('--da_len', int, 128),
            SSL_Argument('--use_rot', str2bool, False),
            SSL_Argument('--lambda_u', float, 1.),
            SSL_Argument('--lambda_reg', float, 1.),
            SSL_Argument('--lambda_mi', float, 1.),
            SSL_Argument('--lambda_kd', float, 1.),
            SSL_Argument('--num_heads', int, 10),
            SSL_Argument('--proj_dim', int, 128),
            SSL_Argument('--use_joint', str2bool, True),
            SSL_Argument('--mask_type', str, 'soft'),
            SSL_Argument('--start_fix', int, 0),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--c_cutoff', float, 0.95),
            SSL_Argument('--ema_alpha', float, 0.9),
            SSL_Argument('--temp_d', float, 1.),
            SSL_Argument('--temp_w', float, 1.),
            SSL_Argument('--temp_s', float, 0.1),
            SSL_Argument('--K', int, 256),

            SSL_Argument('--use_surgery', str2bool, True),
            SSL_Argument('--surgery_mode', str, 'vlr'),
            SSL_Argument('--surgery_scope', str, 'backbone'),
            SSL_Argument('--subspace_dim', int, 10),
            SSL_Argument('--subspace_update_interval', int, 1),
            SSL_Argument('--surgery_alpha', float, 1.0),
        ]
