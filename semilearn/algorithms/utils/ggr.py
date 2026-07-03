import numpy as np
import torch


VALID_SURGERY_MODES = {'vlr', 'osr', 'csr', 'pcgrad_sym', 'conflict_drop'}

_LEGACY_ALIASES = {
    'pcgrad': 'vlr',
    'pcgrad_asym': 'vlr',
    'subspace_orth': 'osr',
    'subspace_signed': 'csr',
    'pcgrad_symmetric': 'pcgrad_sym',
    'pcgrad_standard': 'pcgrad_sym',
    'conflictdrop': 'conflict_drop',
    'drop_aux': 'conflict_drop',
}


def normalize_surgery_mode(mode):
    if mode is None:
        return 'vlr'
    m = str(mode).lower()
    m = _LEGACY_ALIASES.get(m, m)
    if m not in VALID_SURGERY_MODES:
        raise ValueError(
            f"Unknown surgery_mode '{mode}'. Expected one of "
            f"{sorted(VALID_SURGERY_MODES)}."
        )
    return m


def flatten_grads_dense(params, grads, device):
    flat = []
    shapes = []
    for p, g in zip(params, grads):
        shapes.append(p.shape)
        if g is None:
            flat.append(torch.zeros_like(p, device=device).contiguous().view(-1))
        else:
            flat.append(g.to(device).contiguous().view(-1))
    if len(flat) == 0:
        return None, shapes
    return torch.cat(flat, dim=0), shapes


def unflatten_like_dense(vec, shapes, device):
    if vec is None:
        return [None for _ in shapes]
    grads = []
    off = 0
    for shp in shapes:
        n = int(np.prod(shp))
        grads.append(vec[off:off + n].view(shp).to(device))
        off += n
    return grads


class GradSubspace:
    def __init__(self, max_dim=10, eps=1e-12, device='cpu'):
        self.max_dim = int(max_dim)
        self.eps = float(eps)
        self.U = None
        self.device = torch.device(device)

    @torch.no_grad()
    def update(self, v):
        if v is None:
            return
        v = v.detach()
        if v.device != self.device:
            v = v.to(self.device)
        nv = torch.norm(v).clamp_min(self.eps)
        v = v / nv

        if self.U is None:
            self.U = v.unsqueeze(1)
            return

        U = self.U
        proj = U @ (U.t() @ v)
        v_orth = v - proj
        nv = torch.norm(v_orth).clamp_min(self.eps)
        v_orth = v_orth / nv

        self.U = torch.cat([U, v_orth.unsqueeze(1)], dim=1)
        if self.U.shape[1] > self.max_dim:
            self.U = self.U[:, -self.max_dim:]

        self.U, _ = torch.linalg.qr(self.U, mode='reduced')

    @torch.no_grad()
    def project_csr(self, g):
        if (self.U is None) or (g is None):
            return g
        U = self.U
        a = U.t() @ g
        a = torch.minimum(a, torch.zeros_like(a))
        return g - U @ a

    @torch.no_grad()
    def project_osr(self, g):
        if (self.U is None) or (g is None):
            return g
        U = self.U
        a = U.t() @ g
        return g - U @ a


@torch.no_grad()
def _halfspace_project(g_src_vec, g_ref_vec, alpha=1.0):
    if g_src_vec is None:
        return None
    if g_ref_vec is None:
        return g_src_vec
    dot = torch.dot(g_src_vec, g_ref_vec)
    if dot >= 0:
        return g_src_vec
    denom = torch.dot(g_ref_vec, g_ref_vec).clamp_min(1e-12)
    return g_src_vec - float(alpha) * (dot / denom) * g_ref_vec


@torch.no_grad()
def rectify_pair(g_sup_vec, g_aux_vec, mode, grad_subspace=None,
                 subspace_dim=0, surgery_alpha=1.0):
    mode = normalize_surgery_mode(mode)

    if mode == 'vlr':
        return g_sup_vec, _halfspace_project(g_aux_vec, g_sup_vec, alpha=surgery_alpha)

    if mode == 'pcgrad_sym':
        g_sup_safe = _halfspace_project(g_sup_vec, g_aux_vec, alpha=surgery_alpha)
        g_aux_safe = _halfspace_project(g_aux_vec, g_sup_vec, alpha=surgery_alpha)
        return g_sup_safe, g_aux_safe

    if mode == 'conflict_drop':
        if g_aux_vec is None:
            return g_sup_vec, None
        if g_sup_vec is None:
            return g_sup_vec, g_aux_vec
        if torch.dot(g_sup_vec, g_aux_vec) < 0:
            return g_sup_vec, torch.zeros_like(g_aux_vec)
        return g_sup_vec, g_aux_vec

    if (subspace_dim <= 0) or (grad_subspace is None):
        return g_sup_vec, g_aux_vec

    if mode == 'osr':
        return g_sup_vec, grad_subspace.project_osr(g_aux_vec)
    if mode == 'csr':
        return g_sup_vec, grad_subspace.project_csr(g_aux_vec)

    return g_sup_vec, g_aux_vec


def select_surgery_params(model, surgery_scope):
    all_params = [p for p in model.parameters() if p.requires_grad]
    if surgery_scope == 'all':
        return all_params, [True] * len(all_params)

    backbone_ids = set()
    head_ids = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone_ids.add(id(p))
        else:
            head_ids.add(id(p))

    if surgery_scope == 'backbone':
        mask = [(id(p) in backbone_ids) for p in all_params]
    elif surgery_scope == 'head':
        mask = [(id(p) in head_ids) for p in all_params]
    else:
        mask = [True] * len(all_params)

    if not any(mask):
        mask = [True] * len(all_params)
    return all_params, mask
