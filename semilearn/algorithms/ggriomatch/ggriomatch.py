import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import (
    ce_loss, SSL_Argument, str2bool,
    flatten_grads_dense, unflatten_like_dense,
    GradSubspace, rectify_pair, select_surgery_params, normalize_surgery_mode,
)

from .utils import mb_sup_loss, masked_soft_ce


class GGRIOMatchNet(nn.Module):
    def __init__(self, base, num_classes, proj_size=128, use_rot=False):
        super().__init__()
        self.backbone = base
        self.feat_planes = base.num_features
        self.use_rot = use_rot

        self.mlp_proj = nn.Sequential(
            nn.Linear(self.feat_planes, self.feat_planes),
            nn.ReLU(inplace=False),
            nn.Linear(self.feat_planes, proj_size),
        )

        self.mb_classifiers = nn.Linear(proj_size, num_classes * 2, bias=False)
        self.openset_classifier = nn.Linear(proj_size, num_classes + 1)

        if self.use_rot:
            self.rot_classifier = nn.Linear(self.feat_planes, 4, bias=False)
            nn.init.xavier_normal_(self.rot_classifier.weight.data)

        nn.init.xavier_normal_(self.mb_classifiers.weight.data)
        nn.init.xavier_normal_(self.openset_classifier.weight.data)
        self.openset_classifier.bias.data.zero_()

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = self.mlp_proj(feat)

        logits_open = self.openset_classifier(feat_proj)
        logits_mb = self.mb_classifiers(feat_proj)

        out = {
            'feat': feat,
            'feat_proj': feat_proj,
            'logits': logits,
            'logits_open': logits_open,
            'logits_mb': logits_mb,
        }

        if self.use_rot:
            out['logits_rot'] = self.rot_classifier(feat)

        return out

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@ALGORITHMS.register('ggriomatch')
class GGRIOMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

        self.dist_align = args.dist_align
        self.use_rot = args.use_rot
        self.p_cutoff = args.p_cutoff
        self.q_cutoff = args.q_cutoff
        self.lambda_mb = args.mb_loss_ratio
        self.lambda_op = args.op_loss_ratio

        self.use_surgery = args.use_surgery
        self.surgery_mode = normalize_surgery_mode(args.surgery_mode)
        self.surgery_scope = args.surgery_scope
        self.subspace_dim = args.subspace_dim
        self.subspace_update_interval = max(1, args.subspace_update_interval)
        self.surgery_alpha = args.surgery_alpha

        self.use_graph = args.use_graph
        self.lambda_g = args.lambda_g
        self.graph_t = args.graph_t
        self.g_cutoff = args.g_cutoff
        self.graph_use_inmask = args.graph_use_inmask

        self.norm_mode = args.norm_mode

        self.start_unsup_iter = args.start_unsup_iter
        self.rampup_iter = args.rampup_iter

        self.grad_subspace = None
        self._iter = 0

    def set_hooks(self):
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),
            "DistAlignHook"
        )
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = GGRIOMatchNet(
            model, num_classes=self.num_classes, use_rot=self.args.use_rot,
            proj_size=self.args.proj_size
        )
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = GGRIOMatchNet(
            ema_model, num_classes=self.num_classes, use_rot=self.args.use_rot,
            proj_size=self.args.proj_size
        )
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_model_ptr'] = self.hooks_dict['DistAlignHook'].p_model_ptr.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_model_ptr = checkpoint['p_model_ptr'].cuda(self.args.gpu)
        return checkpoint

    def _use_manual_ddp_update(self):
        return (
            self.use_surgery
            and self.distributed
            and isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        )

    def _sync_grad_list(self, params, grads, device):
        if (not self.distributed) or (not dist.is_available()) or (not dist.is_initialized()):
            return grads

        flat_grad, shapes = flatten_grads_dense(params, grads, device=device)
        used_flags = torch.tensor(
            [0.0 if g is None else 1.0 for g in grads],
            device=device,
            dtype=torch.float32,
        )

        if flat_grad is not None:
            dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
            flat_grad.div_(float(self.world_size))
        dist.all_reduce(used_flags, op=dist.ReduceOp.SUM)

        synced_dense = unflatten_like_dense(flat_grad, shapes, device=device)
        synced_grads = []
        for used, g in zip(used_flags.tolist(), synced_dense):
            synced_grads.append(None if used == 0.0 else g)
        return synced_grads

    def _apply_gradients(self, params, grads):
        self.optimizer.zero_grad()

        scale = float(self.loss_scaler.get_scale()) if self.use_amp else 1.0
        for p, g in zip(params, grads):
            if g is None:
                p.grad = None
                continue
            grad = g.detach()
            if scale != 1.0:
                grad = grad * scale
            p.grad = grad.to(dtype=p.dtype)

        if self.use_amp:
            self.loss_scaler.unscale_(self.optimizer)
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, self.clip_grad)
            self.loss_scaler.step(self.optimizer)
            self.loss_scaler.update()
        else:
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, self.clip_grad)
            self.optimizer.step()

        self.scheduler.step()
        self.model.zero_grad()

    def _rectify_pair(self, g_sup_vec, g_aux_vec):
        return rectify_pair(
            g_sup_vec, g_aux_vec,
            mode=self.surgery_mode,
            grad_subspace=self.grad_subspace,
            subspace_dim=self.subspace_dim,
            surgery_alpha=self.surgery_alpha,
        )

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, **kwargs):
        self._iter += 1
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        device = y_lb.device
        use_manual_ddp_update = self._use_manual_ddp_update()
        forward_model = self.model.module if use_manual_ddp_update else self.model

        if self._iter < self.start_unsup_iter:
            lambda_u_t = 0.0
        elif self.rampup_iter <= 0:
            lambda_u_t = float(self.lambda_u)
        else:
            t = (self._iter - self.start_unsup_iter) / float(self.rampup_iter)
            t = max(0.0, min(1.0, t))
            lambda_u_t = float(self.lambda_u) * t

        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s), dim=0)
                outputs = forward_model(inputs)

                logits_all = outputs['logits']
                logits_open_all = outputs['logits_open']
                logits_mb_all = outputs['logits_mb']

                logits_x_lb = logits_all[:num_lb]
                logits_mb_x_lb = logits_mb_all[:num_lb]

                logits_x_ulb_w, logits_x_ulb_s = logits_all[num_lb:].chunk(2)
                logits_open_x_ulb_w, logits_open_x_ulb_s = logits_open_all[num_lb:].chunk(2)
                logits_mb_x_ulb_w, _ = logits_mb_all[num_lb:].chunk(2)

                feat_all = outputs['feat_proj']
                feat_ulb_w, feat_ulb_s = feat_all[num_lb:].chunk(2)
            else:
                raise ValueError("Bad configuration: use_cat should be True!")

            sup_closed_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
            sup_mb_loss = self.lambda_mb * mb_sup_loss(logits_mb_x_lb, y_lb)
            sup_loss = sup_closed_loss + sup_mb_loss

            if self.use_rot:
                x_ulb_r = torch.cat([torch.rot90(x_ulb_w[:num_lb], i, [2, 3]) for i in range(4)], dim=0)
                y_ulb_r = torch.cat([torch.empty(x_ulb_w[:num_lb].size(0)).fill_(i).long()
                                     for i in range(4)], dim=0).to(device)
                self.bn_controller.freeze_bn(forward_model)
                logits_rot = forward_model(x_ulb_r)['logits_rot']
                self.bn_controller.unfreeze_bn(forward_model)
                rot_loss = ce_loss(logits_rot, y_ulb_r, reduction='mean')
            else:
                rot_loss = torch.zeros((), device=device)

        with torch.no_grad():
            p = torch.softmax(logits_x_ulb_w, dim=-1)
            targets_p = p.detach()
            if self.dist_align:
                targets_p = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=targets_p)

            logits_mb = logits_mb_x_ulb_w.view(num_ulb, 2, -1)
            r = torch.softmax(logits_mb, dim=1)
            tmp_range = torch.arange(0, num_ulb, device=device, dtype=torch.long)
            out_scores = torch.sum(targets_p * r[tmp_range, 0, :], dim=1)
            in_mask = (out_scores < 0.5)

            o_neg = r[tmp_range, 0, :]
            o_pos = r[tmp_range, 1, :]
            q = torch.zeros((num_ulb, self.num_classes + 1), device=device)
            q[:, :self.num_classes] = targets_p * o_pos
            q[:, self.num_classes] = torch.sum(targets_p * o_neg, dim=1)
            targets_q = q.detach()

        p_mask = self.call_hook("masking", "MaskingHook", cutoff=self.p_cutoff, logits_x_ulb=targets_p, softmax_x_ulb=False)
        q_mask = self.call_hook("masking", "MaskingHook", cutoff=self.q_cutoff, logits_x_ulb=targets_q, softmax_x_ulb=False)
        p_mask = p_mask.float()
        q_mask = q_mask.float()

        ui_weight = in_mask.float() * p_mask
        op_weight = q_mask

        with self.amp_cm():
            ui_loss = masked_soft_ce(logits_x_ulb_s, targets_p, ui_weight, norm_mode=self.norm_mode)
            op_loss = masked_soft_ce(logits_open_x_ulb_s, targets_q, op_weight, norm_mode=self.norm_mode)

            if self.epoch == 0:
                op_loss = op_loss * 0.0

            if self.use_graph and (self.lambda_g > 0):
                z_w = F.normalize(feat_ulb_w, dim=-1)
                z_s = F.normalize(feat_ulb_s, dim=-1)
                sim = torch.exp(torch.mm(z_w, z_s.t()) / max(self.graph_t, 1e-6))
                sim_probs = sim / (sim.sum(1, keepdim=True).clamp_min(1e-12))

                Q = torch.mm(targets_p, targets_p.t())
                Q.fill_diagonal_(1.0)
                pos_mask = (Q >= float(self.g_cutoff)).float()

                if self.graph_use_inmask:
                    gm = (in_mask.unsqueeze(0) == in_mask.unsqueeze(1)).float()
                else:
                    gm = torch.ones_like(Q)

                Q = Q * pos_mask * gm
                Q = Q / (Q.sum(1, keepdim=True).clamp_min(1e-12))

                loss_g = -(torch.log(sim_probs.clamp_min(1e-12)) * Q).sum(1).mean()
            else:
                loss_g = torch.zeros((), device=device)

            unsup_loss = lambda_u_t * ui_loss
            op_loss_w = self.lambda_op * op_loss
            graph_loss = self.lambda_g * loss_g

            aux_loss = unsup_loss + op_loss_w + graph_loss + rot_loss
            total_loss = sup_loss + aux_loss

        no_aux_update = (lambda_u_t == 0.0 and self.lambda_op == 0.0 and self.lambda_g == 0.0 and (not self.use_rot))
        if use_manual_ddp_update:
            all_params = [p for p in forward_model.parameters() if p.requires_grad]

            if no_aux_update:
                g_final = torch.autograd.grad(
                    total_loss, all_params, retain_graph=False, create_graph=False, allow_unused=True
                )
                g_final = self._sync_grad_list(all_params, g_final, device=device)
                self._apply_gradients(all_params, g_final)
            else:
                all_params, surgery_mask = select_surgery_params(forward_model, self.surgery_scope)

                g_sup_all = torch.autograd.grad(sup_loss, all_params, retain_graph=True, create_graph=False, allow_unused=True)
                g_aux_all = torch.autograd.grad(aux_loss, all_params, retain_graph=False, create_graph=False, allow_unused=True)

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
                        gs = next(safe_sup_it)
                        gu = next(safe_aux_it)
                    if gs is None:
                        g = gu
                    elif gu is None:
                        g = gs
                    else:
                        g = gs + gu
                    g_final.append(g)

                g_final = self._sync_grad_list(all_params, g_final, device=device)
                self._apply_gradients(all_params, g_final)
        elif (not self.use_surgery) or no_aux_update:
            self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)
        else:
            all_params, surgery_mask = select_surgery_params(forward_model, self.surgery_scope)

            g_sup_all = torch.autograd.grad(sup_loss, all_params, retain_graph=True, create_graph=False, allow_unused=True)
            g_aux_all = torch.autograd.grad(aux_loss, all_params, retain_graph=False, create_graph=False, allow_unused=True)

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
                    gs = next(safe_sup_it)
                    gu = next(safe_aux_it)
                if gs is None:
                    g = gu
                elif gu is None:
                    g = gs
                else:
                    g = gs + gu
                g_final.append(g)

            surrogate = torch.zeros((), device=device, dtype=torch.float32)
            for p, g in zip(all_params, g_final):
                if g is None:
                    continue
                surrogate = surrogate + torch.sum(p * g.detach())
            self.call_hook("param_update", "ParamUpdateHook", loss=surrogate)

        with torch.no_grad():
            conf_p = targets_p.max(dim=-1)[0]
            conf_q = targets_q.max(dim=-1)[0]
            tb = {
                'train/sup_closed_loss': float(sup_closed_loss.detach().cpu()),
                'train/sup_mb_loss': float(sup_mb_loss.detach().cpu()),
                'train/sup_loss': float(sup_loss.detach().cpu()),
                'train/ui_loss': float(ui_loss.detach().cpu()),
                'train/op_loss': float(op_loss.detach().cpu()),
                'train/graph_loss': float(loss_g.detach().cpu()),
                'train/rot_loss': float(rot_loss.detach().cpu()),
                'train/unsup_loss': float(unsup_loss.detach().cpu()),
                'train/op_loss_w': float(op_loss_w.detach().cpu()),
                'train/total_loss': float(total_loss.detach().cpu()),
                'train/in_mask_ratio': float(in_mask.float().mean().cpu()),
                'train/p_mask_ratio': float(p_mask.mean().cpu()),
                'train/q_mask_ratio': float(q_mask.mean().cpu()),
                'train/ui_weight_mean': float(ui_weight.mean().cpu()),
                'train/op_weight_mean': float(op_weight.mean().cpu()),
                'train/p_conf_mean': float(conf_p.mean().cpu()),
                'train/q_conf_mean': float(conf_q.mean().cpu()),
                'train/lambda_u_t': float(lambda_u_t),
            }
        return tb

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--dist_align', str2bool, False),
            SSL_Argument('--use_rot', str2bool, False),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--q_cutoff', float, 0.5),
            SSL_Argument('--da_len', int, 128),
            SSL_Argument('--mb_loss_ratio', float, 1.0),
            SSL_Argument('--op_loss_ratio', float, 1.0),
            SSL_Argument('--proj_size', int, 128),

            SSL_Argument('--use_surgery', str2bool, True),
            SSL_Argument('--surgery_mode', str, 'vlr'),
            SSL_Argument('--surgery_scope', str, 'backbone'),
            SSL_Argument('--subspace_dim', int, 0),
            SSL_Argument('--subspace_update_interval', int, 1),
            SSL_Argument('--surgery_alpha', float, 1.0),

            SSL_Argument('--use_graph', str2bool, False),
            SSL_Argument('--lambda_g', float, 1.0),
            SSL_Argument('--graph_t', float, 0.2),
            SSL_Argument('--g_cutoff', float, 0.8),
            SSL_Argument('--graph_use_inmask', str2bool, True),

            SSL_Argument('--norm_mode', str, 'mean'),

            SSL_Argument('--start_unsup_iter', int, 0),
            SSL_Argument('--rampup_iter', int, 0),
        ]
