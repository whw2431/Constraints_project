#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pen_ddf_ppo_sdf.py (MODIFIED - FINAL VERSION with separate plots per test case)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym  # type: ignore
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cvxpy as cp  # type: ignore
from cvxpylayers.torch import CvxpyLayer  # type: ignore
from typing import Callable, Dict, Any, List, Tuple, Union
import time
from torch.utils.data import Dataset, DataLoader
import collections

# ------------------------------------------------------------------------------
TOL = 1e-4  # Tolerance for constraint satisfaction

try:
    import cvxpylayers.torch.cvxpylayer as _cvxmod_patch  # type: ignore


    def _patched_to_torch_local(x, dtype, device):
        arr = np.array(x, dtype=np.float32);  # Ensure float32 for PyTorch
        return torch.from_numpy(arr).to(device)


    _cvxmod_patch.to_torch = _patched_to_torch_local
except ImportError:
    pass
except AttributeError:
    pass

# ==============================================================================
# START: DEFINITIONS FOR PRE-TRAINED CONSTRAINT ENCODER AND CFUNC HELPERS
# ==============================================================================
ConstraintFunction = Callable[[torch.Tensor], torch.Tensor]


class ConstraintEmbeddingTransformer(nn.Module):
    def __init__(self, action_dim: int, d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        if action_dim <= 0: raise ValueError(f"action_dim must be positive, got {action_dim}")
        self.action_dim = action_dim;
        self.d_model = d_model
        self.input_token_dim = action_dim + 1
        self.linear_projection = nn.Linear(self.input_token_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        if d_model > 0 and nhead > 0 and d_model % nhead != 0:  # Ensure nhead is compatible
            actual_nhead = 1
        elif d_model == 0:
            actual_nhead = 0
        else:
            actual_nhead = nhead

        if d_model > 0 and actual_nhead > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=actual_nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, activation=F.gelu
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)
        else:
            self.transformer_encoder = nn.Identity()

    def forward(self, constraint_tokens_batch: torch.Tensor) -> torch.Tensor:
        if constraint_tokens_batch.shape[-1] != self.input_token_dim:
            raise ValueError(
                f"Input token dim mismatch. Expected {self.input_token_dim}, got {constraint_tokens_batch.shape[-1]}")
        batch_size = constraint_tokens_batch.shape[0];
        projected_tokens = self.linear_projection(constraint_tokens_batch)
        cls_tokens_expanded = self.cls_token.expand(batch_size, -1, -1)
        S = torch.cat([cls_tokens_expanded, projected_tokens], dim=1)
        transformer_output = self.transformer_encoder(S);
        return transformer_output[:, 0, :]


def cfunc_l2_norm(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor: return torch.linalg.norm(
    x - center.to(x.device), ord=2, dim=-1) - radius


def cfunc_l1_norm(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor: return torch.linalg.norm(
    x - center.to(x.device), ord=1, dim=-1) - radius


def cfunc_linear_halfspace(x: torch.Tensor, a: torch.Tensor, b: float) -> torch.Tensor:
    a_dev = a.to(x.device)
    if x.ndim > 1 and a_dev.ndim == 1:
        return torch.matmul(x, a_dev) - b
    elif x.ndim == 1 and a_dev.ndim == 1:
        if x.shape[-1] == a_dev.shape[0]: return torch.sum(x * a_dev, dim=-1) - b
    elif x.ndim > 1 and a_dev.ndim > 1 and x.shape == a_dev.shape:
        return torch.sum(x * a_dev, dim=-1) - b
    raise ValueError(f"Shape mismatch for linear halfspace: x={x.shape}, a={a_dev.shape}")


def cfunc_box(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    l_d, h_d = low.to(x.device), high.to(x.device);
    # Ensure dimensions match for broadcasting if needed
    if x.ndim > 1 and l_d.ndim == 1:
        l_d = l_d.unsqueeze(0)
    if x.ndim > 1 and h_d.ndim == 1:
        h_d = h_d.unsqueeze(0)

    violations = torch.cat([(x - h_d).unsqueeze(-1), (l_d - x).unsqueeze(-1)], dim=-1)
    max_dim_violations, _ = torch.max(violations, dim=-1)
    overall_max_violation, _ = torch.max(max_dim_violations, dim=-1, keepdim=False)
    return overall_max_violation


def cfunc_polytope(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    A_d, b_d = A.to(x.device), b.to(x.device);
    if x.ndim > 1:
        violations = torch.matmul(x.unsqueeze(1), A_d.transpose(-1, -2)).squeeze(1) - b_d
    else:
        violations = torch.mv(A_d, x) - b_d
    max_violation, _ = torch.max(violations, dim=-1)
    return max_violation


def cfunc_ellipsoid(x: torch.Tensor, c: torch.Tensor, P_i: torch.Tensor) -> torch.Tensor:
    c_d, Pi_d = c.to(x.device), P_i.to(x.device);
    df = x - c_d
    if df.ndim == 1:
        return torch.dot(df, torch.mv(Pi_d, df)) - 1. if Pi_d.ndim > 1 and Pi_d.shape[0] == df.shape[
            0] else Pi_d.squeeze() * (df * df).sum() - 1.
    else:
        if Pi_d.ndim == 2:
            temp = torch.matmul(df, Pi_d)
            return torch.sum(temp * df, dim=-1) - 1.
        elif Pi_d.ndim == 1 and Pi_d.numel() == 1:
            return Pi_d.squeeze() * torch.sum(df * df, dim=-1) - 1.
        else:
            return Pi_d.squeeze() * torch.sum(df * df, dim=-1) - 1.


def cfunc_soc(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor, c_vec: torch.Tensor,
              d_scalar: float) -> torch.Tensor:
    A_d, b_d, c_d = A.to(x.device), b.to(x.device), c_vec.to(x.device)
    if x.ndim == 1:
        Ax_plus_b = torch.mv(A_d, x) + b_d if A_d.numel() > 0 and (A_d.ndim < 2 or A_d.shape[1] == x.shape[0]) else b_d
        cTx_plus_d = torch.dot(c_d, x) + d_scalar if c_d.numel() > 0 and (
                c_d.ndim < 2 or c_d.shape[0] == x.shape[0]) else d_scalar
    else:
        if A_d.numel() > 0 and (A_d.ndim < 2 or A_d.shape[1] == x.shape[-1]):
            Ax_plus_b = torch.matmul(x, A_d.t()) + b_d.unsqueeze(0)
        else:
            Ax_plus_b = b_d.unsqueeze(0).expand(x.shape[0], -1 if b_d.ndim < 2 else b_d.shape[
                -1]) if b_d.numel() > 0 else torch.zeros(x.shape[0], 0 if b_d.ndim < 2 else b_d.shape[-1],
                                                         device=x.device)

        if c_d.numel() > 0 and (c_d.ndim < 2 or c_d.shape[0] == x.shape[-1]):
            cTx_plus_d = torch.matmul(x, c_d) + d_scalar if c_d.ndim == 1 else torch.matmul(x,
                                                                                            c_d.unsqueeze(-1)).squeeze(
                -1) + d_scalar
        else:
            cTx_plus_d = torch.full((x.shape[0],), d_scalar, device=x.device)

    if Ax_plus_b.shape[-1] == 0:
        norm_Ax_plus_b = torch.zeros_like(cTx_plus_d)
    else:
        norm_Ax_plus_b = torch.linalg.norm(Ax_plus_b, ord=2, dim=-1)

    return norm_Ax_plus_b - cTx_plus_d


class CallableOracleWrapper:
    def __init__(self, o: Callable[[torch.Tensor], Any], d: torch.device):
        self.oracle = o;
        self.device = d

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        v = self.oracle(x.to(self.device));
        if not isinstance(v, torch.Tensor):
            v = torch.tensor([v] if not isinstance(v, (list, tuple, np.ndarray)) else v,
                             dtype=torch.float32, device=self.device)
        elif v.device != self.device:
            v = v.to(self.device)

        if v.ndim == 0:
            v = v.unsqueeze(0)
        return v.float()


def build_cfunc_from_raw_data(raw: Dict[str, Any], dev: torch.device) -> ConstraintFunction:
    ct = raw.get('type', 'unknown');
    ps = raw.get('params', {});
    pp = {}
    for k, v_orig in ps.items():
        if k == 'components' and ct == 'union':
            pp[k] = v_orig
        elif isinstance(v_orig, (np.ndarray, list)):
            try:
                v_np = np.array(v_orig, dtype=np.float32)
            except ValueError:
                v_np = np.array(v_orig, dtype=object).astype(np.float32)
            pp[k] = torch.from_numpy(v_np).to(dev)
        elif isinstance(v_orig, torch.Tensor):
            pp[k] = v_orig.to(dtype=torch.float32, device=dev)
        else:
            pp[k] = v_orig

    if ct == 'l2_norm':
        return lambda x: cfunc_l2_norm(x, pp['center'], pp['radius'])
    elif ct == 'l1_norm':
        return lambda x: cfunc_l1_norm(x, pp['center'], pp['radius'])
    elif ct == 'linear_halfspace':
        return lambda x: cfunc_linear_halfspace(x, pp['a'], pp['b'])
    elif ct == 'box':
        return lambda x: cfunc_box(x, pp['low'], pp['high'])
    elif ct == 'polytope':
        return lambda x: cfunc_polytope(x, pp['A'], pp['b'])
    elif ct == 'ellipsoid':
        return lambda x: cfunc_ellipsoid(x, pp['center'], pp.get('P_inv', pp.get('P_i')))
    elif ct == 'soc':
        return lambda x: cfunc_soc(x, pp['A_soc'], pp['b_soc'], pp['c_soc'], pp['d_soc'])
    elif ct == 'union':
        component_cfuncs = [build_cfunc_from_raw_data(comp_raw_data, dev) for comp_raw_data in pp['components']]

        def cfunc_union(x: torch.Tensor) -> torch.Tensor:
            evaluations = [comp_cf(x) for comp_cf in component_cfuncs];
            if not evaluations:
                return torch.full_like(x[:, 0] if x.ndim > 1 else x, float('inf'), device=x.device)

            processed_evals = []
            is_batched_x = x.ndim > 1
            batch_size_x = x.shape[0] if is_batched_x else 1

            for i, ev in enumerate(evaluations):
                cev = ev
                if is_batched_x:
                    if cev.ndim == 0:
                        cev = cev.expand(batch_size_x)
                    elif cev.shape[0] != batch_size_x and cev.numel() == 1:
                        cev = cev.reshape(1).expand(batch_size_x)
                    elif cev.shape[0] != batch_size_x:
                        raise ValueError(
                            f"Union component cfunc {i} output shape {cev.shape} incompatible with batched input x shape {x.shape}")
                else:
                    if cev.numel() != 1:
                        raise ValueError(
                            f"Union component cfunc {i} output {cev.shape} for unbatched input x {x.shape} should be scalar.")
                    if cev.ndim > 0 and cev.numel() == 1: cev = cev.reshape(())
                processed_evals.append(cev)

            if not processed_evals:
                return torch.full_like(x[:, 0] if x.ndim > 1 else x, float('inf'), device=x.device)

            stacked_evals = torch.stack(processed_evals, dim=0)
            min_violations, _ = torch.min(stacked_evals, dim=0)
            return min_violations

        return cfunc_union
    elif callable(raw.get('cfunc_oracle')):
        return CallableOracleWrapper(raw['cfunc_oracle'], dev)
    else:
        raise ValueError(f"Unsupported constraint type: {ct} with parameters: {raw}")


PPO_ENV_ID = 'InvertedPendulum-v4';
_senv = gym.make(PPO_ENV_ID);
ENVIRONMENT_ACTION_DIM = _senv.action_space.shape[0];
ENVIRONMENT_STATE_DIM = _senv.observation_space.shape[0];
_senv.close()
PRETRAINED_ACTION_DIM = ENVIRONMENT_ACTION_DIM
PRETRAINED_D_MODEL = 32;
PRETRAINED_N_HEAD = 1 if PRETRAINED_D_MODEL == 0 else (2 if PRETRAINED_D_MODEL % 2 == 0 else 1);
PRETRAINED_NUM_ENCODER_LAYERS = 2
PRETRAINED_DIM_FEEDFORWARD = PRETRAINED_D_MODEL * 2;
PRETRAINED_DROPOUT = 0.1
PRETRAINED_MODEL_PATH = f"simclr_encoder_action_dim{PRETRAINED_ACTION_DIM}.pth"
PPO_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using PPO_DEVICE: {PPO_DEVICE} ---")

print(f"--- Initializing Constraint Encoder (ActionDim: {PRETRAINED_ACTION_DIM}) ---")
pretrained_constraint_encoder = ConstraintEmbeddingTransformer(action_dim=PRETRAINED_ACTION_DIM,
                                                               d_model=PRETRAINED_D_MODEL, nhead=PRETRAINED_N_HEAD,
                                                               num_encoder_layers=PRETRAINED_NUM_ENCODER_LAYERS,
                                                               dim_feedforward=PRETRAINED_DIM_FEEDFORWARD,
                                                               dropout=PRETRAINED_DROPOUT)
try:
    print(f"Loading pre-trained constraint encoder from: {PRETRAINED_MODEL_PATH}")
    pretrained_constraint_encoder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=PPO_DEVICE))
    print("Successfully loaded pre-trained constraint encoder.")
except Exception as e:
    print(f"ERROR loading constraint encoder from {PRETRAINED_MODEL_PATH}: {e}. Using RANDOM encoder.")
pretrained_constraint_encoder.to(PPO_DEVICE).eval()


def generate_tokens_for_single_constraint(raw: Dict[str, Any], N_s: int, s_rng: Tuple[float, float],
                                          dev: torch.device) -> torch.Tensor:
    cf = build_cfunc_from_raw_data(raw, dev);
    tks: List[torch.Tensor] = [];
    m_s, mx_s = s_rng;
    d_s = mx_s - m_s
    current_action_dim_for_sampling = PRETRAINED_ACTION_DIM if PRETRAINED_ACTION_DIM > 0 else 1

    for _ in range(N_s):
        x_s = (torch.rand(current_action_dim_for_sampling, device=dev) * d_s) + m_s;
        if PRETRAINED_ACTION_DIM == 0: x_s = x_s.squeeze(0)

        c_v = cf(x_s)
        if not isinstance(c_v, torch.Tensor):
            c_v = torch.tensor([c_v], dtype=torch.float32, device=dev)
        if c_v.numel() != 1:
            c_v = c_v.flatten()[0]
        c_i = c_v.reshape(1).float();
        tks.append(torch.cat([x_s.flatten(), c_i]))
    return torch.stack(tks)


def new_encode_constraint(raw: Dict[str, Any], N_s: int, s_rng: Tuple[float, float], dev: torch.device) -> torch.Tensor:
    t_seq = generate_tokens_for_single_constraint(raw, N_s, s_rng, dev)
    with torch.no_grad():
        emb = pretrained_constraint_encoder(t_seq.unsqueeze(0))
    return emb.squeeze(0)


# ==============================================================================
# END: PRE-TRAINED ENCODER
# ==============================================================================

# ##############################################################################
# START: MODIFIED ConvexBall CLASS
# ##############################################################################
class ConvexBall:
    _lc = {};  # Layer cache: (dim, device_str) -> CvxpyLayer

    def __init__(self, c: Union[np.ndarray, List, Tuple], r: float):
        if not isinstance(c, np.ndarray):
            c_np_raw = np.array(c)
        else:
            c_np_raw = c

        if c_np_raw.ndim == 0:
            cnp = c_np_raw.astype(np.float32).reshape(1, )
        elif c_np_raw.ndim == 1:
            cnp = c_np_raw.astype(np.float32)
        else:
            raise ValueError(f"ConvexBall center c must be scalar or 1D, got shape {c_np_raw.shape}")

        self.c = torch.from_numpy(cnp).to(PPO_DEVICE);
        self.r = float(r);
        self.dim = len(cnp)
        if self.dim == 0:
            raise ValueError("ConvexBall dimension cannot be 0 after processing center.")

    @classmethod
    def _ensure_layer(cls, d: int, dev: torch.device):
        ck = (d, str(dev));
        if ck in cls._lc and cls._lc[ck] is not None: return
        if d <= 0: cls._lc[ck] = None; return

        z_var = cp.Variable(d, name="z_decision_var")
        x_param = cp.Parameter(d, name="x_point_to_project")
        c_param = cp.Parameter(d, name="c_ball_center")
        r_param = cp.Parameter(nonneg=True, name="r_ball_radius")

        objective = cp.Minimize(0.5 * cp.sum_squares(z_var - x_param))
        constraints = [cp.norm(z_var - c_param, 2) <= r_param]

        problem = cp.Problem(objective, constraints)
        if not problem.is_dcp(dpp=True):
            pass

        try:
            cls._lc[ck] = CvxpyLayer(problem,
                                     parameters=[x_param, c_param, r_param],
                                     variables=[z_var]
                                     )
        except Exception as e:
            print(f"Error creating CvxpyLayer for ConvexBall (dim {d}, device {dev}): {e}")
            cls._lc[ck] = None;

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            c_on_x_device = self.c.to(x.device)
            low = c_on_x_device - self.r
            high = c_on_x_device + self.r
            return torch.clamp(x, min=low.squeeze(), max=high.squeeze())

        if self.dim <= 0: return x
        is_scalar_input_shape = x.ndim == 1;
        xb = x.unsqueeze(0) if is_scalar_input_shape else x

        if xb.shape[-1] != self.dim:
            raise ValueError(f"ConvexBall (dim {self.dim}) project expects last dim {self.dim}, got shape {xb.shape}")

        current_device = xb.device;
        cache_key = (self.dim, str(current_device));

        if cache_key not in self._lc or self._lc[cache_key] is None:
            try:
                self._ensure_layer(self.dim, current_device)
            except Exception as e_layer:
                print(
                    f"Fallback needed: Error during _ensure_layer for dim {self.dim}, device {current_device}: {e_layer}")

        cvx_layer_instance = self._lc.get(cache_key)

        if cvx_layer_instance is None:
            c_on_current_device = self.c.to(current_device)
            c_expanded = c_on_current_device.expand_as(xb)
            dist_to_center = torch.linalg.norm(xb - c_expanded, dim=-1, keepdim=True)
            points_outside = dist_to_center > self.r
            scale_factor = self.r / (dist_to_center + 1e-9)
            projected_points = c_expanded + (xb - c_expanded) * scale_factor
            zb = torch.where(points_outside, projected_points, xb)
            return zb.squeeze(0) if is_scalar_input_shape else zb

        center_on_current_device = self.c.to(current_device)
        center_expanded = center_on_current_device.expand_as(xb)
        radius_expanded = torch.tensor(self.r, dtype=torch.float32, device=current_device).expand(xb.shape[0])

        try:
            projected_batch, = cvx_layer_instance(xb.float(), center_expanded.float(), radius_expanded)
        except Exception as e_cvx_call:
            print(
                f"Warning: CvxpyLayer call failed during ConvexBall.project (dim {self.dim}, dev {current_device}). Error: {e_cvx_call}. Input x: {xb.detach().cpu().numpy()}. Using geometric fallback.")
            c_expanded_fb = self.c.to(current_device).expand_as(xb)
            dist_to_center_fb = torch.linalg.norm(xb - c_expanded_fb, dim=-1, keepdim=True)
            points_outside_fb = dist_to_center_fb > self.r
            scale_factor_fb = self.r / (dist_to_center_fb + 1e-9)
            projected_points_fb = c_expanded_fb + (xb - c_expanded_fb) * scale_factor_fb
            projected_batch = torch.where(points_outside_fb, projected_points_fb, xb)
            if projected_batch.shape != xb.shape:
                projected_batch = xb

        return projected_batch.squeeze(0) if is_scalar_input_shape else projected_batch

    def violation(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            c_on_x_device = self.c.to(x.device)
            return torch.abs(x - c_on_x_device.squeeze()) - self.r

        if self.dim <= 0:
            return torch.tensor(-self.r if self.r > 0 else 0., device=x.device)

        c_on_x_device = self.c.to(x.device)
        if x.shape[-1] != self.dim:
            print(f"Warning: ConvexBall.violation shape mismatch. x.shape: {x.shape}, self.dim: {self.dim}")
            return torch.full(x.shape[:-1] if x.ndim > 1 else (), float('inf'),
                              device=x.device)
        diff = x - c_on_x_device
        diff_norm = torch.linalg.norm(diff, ord=2, dim=-1)
        violation_val = diff_norm - self.r
        return violation_val


# ##############################################################################
# END: MODIFIED ConvexBall CLASS
# ##############################################################################


class SDFProjector(nn.Module):
    def __init__(self, tau=0.05):
        super().__init__();
        self.tau = tau

    def forward(self, a0: torch.Tensor, sets: list) -> torch.Tensor:
        is_batched_input = a0.ndim > 1;
        a0_batched = a0 if is_batched_input else a0.unsqueeze(0);
        action_dim_of_a0 = a0_batched.shape[-1]

        if not sets: return a0

        phi_evaluations_list = [];
        valid_sets_for_sdf = []

        # ### MODIFICATION START ###
        # Added handling for box constraints which do not have a .dim attribute
        # but have an action_dim in their raw data. For simplicity, we assume
        # all sets passed here are compatible if they don't have a .dim check.
        for i, constraint_set in enumerate(sets):
            is_dim_compatible = False
            if hasattr(constraint_set, 'dim'):
                if constraint_set.dim == action_dim_of_a0:
                    is_dim_compatible = True
            elif isinstance(constraint_set, dict) and constraint_set.get('type') == 'box':
                is_dim_compatible = True  # Assume box is compatible for SDF projection

            if is_dim_compatible:
                try:
                    # For dict-based constraints like our box, we need to build the cfunc
                    if isinstance(constraint_set, dict):
                        cfunc = build_cfunc_from_raw_data(constraint_set, a0.device)
                        violation_values = cfunc(a0_batched)
                    else:  # For object-based constraints like ConvexBall
                        violation_values = constraint_set.violation(a0_batched)

                    phi_evaluations_list.append(violation_values)
                    valid_sets_for_sdf.append(constraint_set)
                except Exception as e:
                    pass
        # ### MODIFICATION END ###

        if not phi_evaluations_list or not valid_sets_for_sdf:
            return a0

        phi_tensor = torch.stack(phi_evaluations_list, dim=0);

        if phi_tensor.numel() == 0:
            return a0

        sdf_phi_value = -self.tau * torch.logsumexp(-phi_tensor / self.tau, dim=0)

        if torch.isnan(sdf_phi_value).any() or torch.isinf(sdf_phi_value).any() or (
                sdf_phi_value.abs().max().item() > 1e6 if sdf_phi_value.numel() > 0 else False):
            min_phi_violations, indices_of_min_sets = torch.min(phi_tensor, dim=0)
            projected_fallback_actions = []
            for item_idx in range(a0_batched.shape[0]):
                action_item = a0_batched[item_idx]
                closest_set_index = indices_of_min_sets[
                    item_idx].item() if indices_of_min_sets.ndim > 0 else indices_of_min_sets.item()
                set_to_project_onto = valid_sets_for_sdf[closest_set_index]

                # Cannot project onto a dict representation, so we skip for box
                if isinstance(set_to_project_onto, dict):
                    projected_fallback_actions.append(action_item)
                    continue

                try:
                    projected_fallback_actions.append(set_to_project_onto.project(action_item))
                except Exception as e_proj_fb:
                    projected_fallback_actions.append(action_item)

            projected_a0 = torch.stack(projected_fallback_actions) if projected_fallback_actions else a0_batched
            return projected_a0.squeeze(0) if not is_batched_input and projected_a0.shape[0] == 1 else projected_a0

        a0_for_grad = a0_batched.clone().detach().requires_grad_(True)

        phi_evals_for_grad = []
        for s_idx, s_set_instance in enumerate(valid_sets_for_sdf):
            # ### MODIFICATION START ###
            # Handle dict-based constraints for gradient calculation too
            if isinstance(s_set_instance, dict):
                cfunc_grad = build_cfunc_from_raw_data(s_set_instance, a0_for_grad.device)
                viol = cfunc_grad(a0_for_grad)
            else:
                viol = s_set_instance.violation(a0_for_grad)
            # ### MODIFICATION END ###
            phi_evals_for_grad.append(viol)

        if not phi_evals_for_grad:
            return a0

        phi_tensor_for_grad = torch.stack(phi_evals_for_grad, dim=0)

        sdf_phi_for_grad = -self.tau * torch.logsumexp(-phi_tensor_for_grad / self.tau, dim=0)

        try:
            target_for_grad = sdf_phi_for_grad.sum() if sdf_phi_for_grad.numel() > 1 else sdf_phi_for_grad
            grad_sdf_wrt_a0, = torch.autograd.grad(
                outputs=target_for_grad, inputs=(a0_for_grad,),
                create_graph=False, retain_graph=False, allow_unused=False
            )
        except RuntimeError as e_grad:
            return a0

        if grad_sdf_wrt_a0 is None:
            return a0

        denominator_squared_norm = grad_sdf_wrt_a0.norm(dim=-1, keepdim=True).pow(2) + 1e-9
        sdf_phi_broadcastable = sdf_phi_for_grad.unsqueeze(
            -1) if sdf_phi_for_grad.ndim < grad_sdf_wrt_a0.ndim else sdf_phi_for_grad
        projected_a0 = a0_batched - (sdf_phi_broadcastable / denominator_squared_norm) * grad_sdf_wrt_a0;

        return projected_a0.squeeze(0) if not is_batched_input and projected_a0.shape[0] == 1 else projected_a0


def hard_project_to_union_of_convex_sets(action_raw: torch.Tensor, convex_component_sets: List,
                                         device: torch.device) -> torch.Tensor:
    if not convex_component_sets: return action_raw

    is_batched_input = action_raw.ndim > 1;
    action_batch_input = action_raw if is_batched_input else action_raw.unsqueeze(0)

    final_projected_actions_list = []

    for i in range(action_batch_input.shape[0]):
        current_single_action = action_batch_input[i].to(device);
        action_dim_of_current = current_single_action.shape[0]

        # ### MODIFICATION START ###
        # This function now needs to handle the box constraint, which will be a dict
        # and doesn't have a .project method. We'll handle it separately.
        compatible_component_sets = []
        for comp_set in convex_component_sets:
            if isinstance(comp_set, ConvexBall) and comp_set.dim == action_dim_of_current:
                compatible_component_sets.append(comp_set)
            elif isinstance(comp_set, dict) and comp_set.get('type') == 'box':
                compatible_component_sets.append(comp_set)  # Add the box dict
        # ### MODIFICATION END ###

        if not compatible_component_sets:
            final_projected_actions_list.append(current_single_action)
            continue

        is_already_inside_a_set = False
        for comp_set in compatible_component_sets:
            try:
                # ### MODIFICATION START ###
                # Build cfunc on the fly for dict-based constraints
                if isinstance(comp_set, dict):
                    cfunc = build_cfunc_from_raw_data(comp_set, current_single_action.device)
                    violation = cfunc(current_single_action)
                else:  # Original logic for ConvexBall objects
                    violation = comp_set.violation(current_single_action)

                if violation.item() <= TOL:
                    final_projected_actions_list.append(current_single_action);
                    is_already_inside_a_set = True;
                    break
                # ### MODIFICATION END ###
            except Exception as e_viol_check:
                pass

        if is_already_inside_a_set:
            continue

        candidate_projections = [];
        squared_distances_to_original = [];
        projection_succeeded_for_any_component = False

        for comp_set in compatible_component_sets:
            try:
                # ### MODIFICATION START ###
                # Handle projection for box constraint
                if isinstance(comp_set, dict) and comp_set.get('type') == 'box':
                    low = torch.tensor(comp_set['params']['low'], device=device)
                    high = torch.tensor(comp_set['params']['high'], device=device)
                    projected_point_onto_comp = torch.clamp(current_single_action, min=low, max=high)
                else:  # Original logic for ConvexBall
                    projected_point_onto_comp = comp_set.project(current_single_action);
                # ### MODIFICATION END ###

                candidate_projections.append(projected_point_onto_comp);
                squared_distances_to_original.append(
                    torch.sum((current_single_action - projected_point_onto_comp) ** 2)
                );
                projection_succeeded_for_any_component = True
            except Exception as e_proj_comp:
                center_str = 'N/A'
                if isinstance(comp_set, ConvexBall):
                    center_str = str(comp_set.c.cpu().numpy())
                elif isinstance(comp_set, dict):
                    center_str = f"Box[{comp_set['params']['low']}, {comp_set['params']['high']}]"
                print(
                    f"Warning: HardProj - Projection failed for component set ({center_str}) with action {current_single_action.cpu().numpy()}. Error: {e_proj_comp}")
                pass

        if not projection_succeeded_for_any_component or not candidate_projections:
            final_projected_actions_list.append(current_single_action);
            continue

        min_dist_idx = torch.argmin(torch.stack(squared_distances_to_original))
        best_projection = candidate_projections[min_dist_idx]
        final_projected_actions_list.append(best_projection)

    if not final_projected_actions_list:
        return action_raw

    final_actions_stacked = torch.stack(final_projected_actions_list)
    return final_actions_stacked.squeeze(0) if not is_batched_input and final_actions_stacked.shape[
        0] == 1 else final_actions_stacked


class CurriculumSampler:
    def __init__(self, action_dim_for_constraints: int):
        self.act_dim = action_dim_for_constraints
        self.levels = [1, 2]
        self.w = {1: 1.0, 2: 1.0}
        self.cover = {1: 0, 2: 0}
        self.viol = {1: 0.0, 2: 0.0}
        self.device = PPO_DEVICE

    def update_weights(self, eta=0.1, eps=1e-6, difficulty_lambda=1.0):
        for l in self.levels:
            coverage_score = 1.0 / (self.cover[l] + eps)
            avg_correction_difficulty = self.viol[l] / max(1, self.cover[l])
            val = coverage_score + difficulty_lambda * avg_correction_difficulty
            self.w[l] = (1 - eta) * self.w[l] + eta * val

    def record_difficulty_metric(self, level: int, correction_metric_value: float):
        self.viol[level] = self.viol.get(level, 0.0) + correction_metric_value

    def sample(self) -> Dict[str, Any]:
        total_weight = sum(self.w.values())
        if total_weight <= 0:
            probabilities = np.ones(len(self.levels)) / len(self.levels)
        else:
            probabilities = np.array([self.w[l] / total_weight for l in self.levels])

        lvl = np.random.choice(self.levels, p=probabilities)
        self.cover[lvl] += 1

        raw_data = {'level': lvl, 'action_dim': self.act_dim}
        act_dim_to_use = max(1, self.act_dim)

        if lvl == 1:
            c_np = np.random.uniform(-0.8, 0.8, act_dim_to_use).astype(np.float32)
            if self.act_dim == 0: c_np = c_np.squeeze()
            r_val = np.random.uniform(0.4, 0.8)
            raw_data.update(
                {'name': f'L1_Ball_R{r_val:.1f}', 'type': 'l2_norm',
                 'params': {'center': c_np if self.act_dim > 0 else np.array([c_np.item()]), 'radius': r_val},
                 'convex': True,
                 'sets': [ConvexBall(c_np if self.act_dim > 0 else np.array([c_np.item()]), r_val)]})
        else:
            c1_np = np.random.uniform(-1.2, -0.2, act_dim_to_use).astype(np.float32)
            if self.act_dim == 0: c1_np = c1_np.squeeze()
            c2_np = np.random.uniform(0.2, 1.2, act_dim_to_use).astype(np.float32)
            if self.act_dim == 0: c2_np = c2_np.squeeze()
            r_val = np.random.uniform(0.2, 0.4)

            c1_param = c1_np if self.act_dim > 0 else np.array([c1_np.item()])
            c2_param = c2_np if self.act_dim > 0 else np.array([c2_np.item()])

            comp1 = {'type': 'l2_norm', 'params': {'center': c1_param, 'radius': r_val}, 'action_dim': self.act_dim}
            comp2 = {'type': 'l2_norm', 'params': {'center': c2_param, 'radius': r_val}, 'action_dim': self.act_dim}
            raw_data.update({'name': f'L2_Union2Balls_R{r_val:.1f}', 'type': 'union',
                             'params': {'components': [comp1, comp2]},
                             'convex': False,
                             'sets': [ConvexBall(c1_param, r_val), ConvexBall(c2_param, r_val)]})
        return raw_data


class StateConstraintEmbedder(nn.Module):
    def __init__(self, state_dim: int, constraint_embedding_dim: int, output_embed_dim: int, dropout: float = 0.1):
        super().__init__();
        self.fc_fusion = nn.Linear(state_dim + constraint_embedding_dim, output_embed_dim);
        self.activation = nn.Tanh()

    def forward(self, s: torch.Tensor, e_constraint: torch.Tensor) -> torch.Tensor:
        s_batched = s if s.ndim > 1 else s.unsqueeze(0)
        e_constraint_batched = e_constraint if e_constraint.ndim > 1 else e_constraint.unsqueeze(0)

        if s_batched.shape[0] != e_constraint_batched.shape[0] and e_constraint_batched.shape[0] == 1:
            e_constraint_batched = e_constraint_batched.expand(s_batched.shape[0], -1)

        combined_input = torch.cat([s_batched.float(), e_constraint_batched.float()], dim=-1)
        fused_embedding = self.fc_fusion(combined_input)
        activated_embedding = self.activation(fused_embedding)
        return activated_embedding


class HardProjPolicy(nn.Module):
    def __init__(self, ppo_joint_embed_dim, action_dim_out, hid_size=64):
        super().__init__();
        self.actor_fc = nn.Sequential(
            nn.Linear(ppo_joint_embed_dim, hid_size), nn.Tanh(),
            nn.Linear(hid_size, hid_size), nn.Tanh()
        )
        self.mu = nn.Linear(hid_size, action_dim_out);
        self.log_std = nn.Parameter(torch.full((action_dim_out,), np.log(0.3), dtype=torch.float32));

        self.critic_fc = nn.Sequential(
            nn.Linear(ppo_joint_embed_dim, hid_size), nn.Tanh(),
            nn.Linear(hid_size, hid_size), nn.Tanh()
        )
        self.value_out = nn.Linear(hid_size, 1)

    def forward(self, joint_embedding_et):
        et_batched = joint_embedding_et if joint_embedding_et.ndim > 1 else joint_embedding_et.unsqueeze(0)

        actor_features = self.actor_fc(et_batched)
        action_mean = self.mu(actor_features)
        action_std = torch.exp(self.log_std).expand_as(action_mean)

        critic_features = self.critic_fc(et_batched)
        state_value = self.value_out(critic_features).squeeze(-1)

        if joint_embedding_et.ndim == 1 and action_mean.shape[0] == 1:
            return action_mean.squeeze(0), action_std.squeeze(0), state_value.squeeze(0)
        return action_mean, action_std, state_value


class LagrangePolicy(nn.Module):
    def __init__(self, ppo_joint_embed_dim, action_dim_out, hid_size=64):
        super().__init__();
        self.actor_fc = nn.Sequential(
            nn.Linear(ppo_joint_embed_dim, hid_size), nn.Tanh(),
            nn.Linear(hid_size, hid_size), nn.Tanh()
        )
        self.mu = nn.Linear(hid_size, action_dim_out);
        self.log_std = nn.Parameter(torch.full((action_dim_out,), np.log(0.3), dtype=torch.float32));

        self.critic_fc = nn.Sequential(
            nn.Linear(ppo_joint_embed_dim, hid_size), nn.Tanh(),
            nn.Linear(hid_size, hid_size), nn.Tanh()
        )
        self.value_out = nn.Linear(hid_size, 1)

    def forward(self, joint_embedding_et):
        et_batched = joint_embedding_et if joint_embedding_et.ndim > 1 else joint_embedding_et.unsqueeze(0)

        actor_features = self.actor_fc(et_batched)
        action_mean = self.mu(actor_features)
        action_std = torch.exp(self.log_std).expand_as(action_mean)

        critic_features = self.critic_fc(et_batched)
        state_value = self.value_out(critic_features).squeeze(-1)

        if joint_embedding_et.ndim == 1 and action_mean.shape[0] == 1:
            return action_mean.squeeze(0), action_std.squeeze(0), state_value.squeeze(0)
        return action_mean, action_std, state_value


class PlainPPOPolicy(nn.Module):
    def __init__(self, state_dim_in, action_dim_out, hid_size=64):
        super().__init__();
        self.actor_fc = nn.Sequential(
            nn.Linear(state_dim_in, hid_size), nn.Tanh(),
            nn.Linear(hid_size, hid_size), nn.Tanh()
        )
        self.mu = nn.Linear(hid_size, action_dim_out);
        self.log_std = nn.Parameter(torch.full((action_dim_out,), np.log(0.3), dtype=torch.float32));

        self.critic_fc = nn.Sequential(
            nn.Linear(state_dim_in, hid_size), nn.Tanh(),
            nn.Linear(hid_size, hid_size), nn.Tanh()
        )
        self.value_out = nn.Linear(hid_size, 1)

    def forward(self, state_s):
        s_batched = state_s.float() if state_s.ndim > 1 else state_s.float().unsqueeze(0)

        actor_features = self.actor_fc(s_batched)
        action_mean = self.mu(actor_features)
        action_std = torch.exp(self.log_std).expand_as(action_mean)

        critic_features = self.critic_fc(s_batched)
        state_value = self.value_out(critic_features).squeeze(-1)

        if state_s.ndim == 1 and action_mean.shape[0] == 1:
            return action_mean.squeeze(0), action_std.squeeze(0), state_value.squeeze(0)
        return action_mean, action_std, state_value


def gaussian_logp(action_a, mean_mu, std_dev_std):
    std_stable = std_dev_std.clamp(min=1e-5)
    term1_sq_err = ((action_a - mean_mu) / std_stable).pow(2)
    term2_log_std = 2 * torch.log(std_stable)
    term3_log_2pi = np.log(2 * np.pi)

    log_prob_per_dim = -0.5 * (term1_sq_err + term2_log_std + term3_log_2pi)
    return torch.sum(log_prob_per_dim, dim=-1)


def finish_episode(ep_b, gamma=0.99, lam=0.90):
    R, A, nv = 0., 0., 0.
    for i in reversed(range(len(ep_b))):
        s, a, lp, r, v, cf_flag = ep_b[i];
        R = r + gamma * R;
        delta = r + gamma * nv - v;
        A = delta + gamma * lam * A;
        ep_b[i] = (s, a, lp, R, A, cf_flag);
        nv = v


def ppo_update(buffer_data, policy_to_update, ppo_optimizer, clip_epsilon, num_ppo_epochs, mini_batch_size,
               is_plain_ppo_policy=False, use_penalty_lagrangian=False,
               current_lambda_penalty_val=1.0, learning_rate_for_lambda=1e-4):
    if not buffer_data: return current_lambda_penalty_val

    s_or_et_list, actions_list, old_log_probs_list, returns_list, advantages_list, cost_flags_list = zip(*buffer_data)

    actions_batch = torch.stack(actions_list).to(PPO_DEVICE)
    if not actions_list:
        return current_lambda_penalty_val

    old_log_probs_batch = torch.stack(old_log_probs_list).to(PPO_DEVICE);
    returns_batch = torch.tensor(returns_list, dtype=torch.float32, device=PPO_DEVICE)

    advantages_tensor = torch.tensor(advantages_list, dtype=torch.float32, device=PPO_DEVICE);
    if advantages_tensor.numel() > 1:
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std()
        normalized_advantages_batch = (advantages_tensor - adv_mean) / (adv_std + 1e-8)
    else:
        normalized_advantages_batch = advantages_tensor

    cost_flags_batch = torch.tensor(cost_flags_list, dtype=torch.float32,
                                    device=PPO_DEVICE) if use_penalty_lagrangian else torch.zeros_like(
        normalized_advantages_batch)

    if is_plain_ppo_policy:
        state_tensors_list = [torch.as_tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s.float()
                              for s in s_or_et_list]
        policy_input_batch = torch.stack(state_tensors_list).to(PPO_DEVICE)
    else:
        valid_embeddings = [item for item in s_or_et_list if item is not None and item.numel() > 0]
        if not valid_embeddings: return current_lambda_penalty_val
        try:
            policy_input_batch = torch.stack(valid_embeddings).to(PPO_DEVICE)
        except RuntimeError as e_stack_emb:
            print(f"PPO Update: Error stacking policy embeddings: {e_stack_emb}. Embeddings: {valid_embeddings}")
            return current_lambda_penalty_val

    min_batch_len = min(len(policy_input_batch), len(actions_batch), len(old_log_probs_batch), len(returns_batch),
                        len(normalized_advantages_batch), len(cost_flags_batch))
    if min_batch_len == 0: return current_lambda_penalty_val

    policy_input_batch = policy_input_batch[:min_batch_len]
    actions_batch = actions_batch[:min_batch_len]
    old_log_probs_batch = old_log_probs_batch[:min_batch_len]
    returns_batch = returns_batch[:min_batch_len]
    normalized_advantages_batch = normalized_advantages_batch[:min_batch_len]
    cost_flags_batch = cost_flags_batch[:min_batch_len]

    if policy_input_batch.shape[0] == 0: return current_lambda_penalty_val

    ppo_dataset = torch.utils.data.TensorDataset(policy_input_batch, actions_batch, old_log_probs_batch, returns_batch,
                                                 normalized_advantages_batch, cost_flags_batch)
    actual_mini_batch_size = min(mini_batch_size, len(ppo_dataset))
    if actual_mini_batch_size == 0: return current_lambda_penalty_val

    ppo_loader = DataLoader(ppo_dataset, batch_size=actual_mini_batch_size, shuffle=True)

    for _epoch in range(num_ppo_epochs):
        for p_in_mb, a_mb, lp_old_mb, r_mb, adv_mb, cost_mb in ppo_loader:
            mu_new_mb, std_new_mb, val_new_mb = policy_to_update(p_in_mb);
            lp_new_mb = gaussian_logp(a_mb, mu_new_mb, std_new_mb);
            ratio_mb = torch.exp(lp_new_mb - lp_old_mb)
            adv_effective_mb = adv_mb - current_lambda_penalty_val * cost_mb if use_penalty_lagrangian else adv_mb
            surr1_mb = ratio_mb * adv_effective_mb;
            surr2_mb = torch.clamp(ratio_mb, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_effective_mb
            actor_loss_mb = -torch.mean(torch.min(surr1_mb, surr2_mb));
            critic_loss_mb = F.mse_loss(val_new_mb, r_mb)
            entropy_mb = torch.mean(torch.sum(0.5 * torch.log(2 * np.pi * np.e * (std_new_mb.pow(2) + 1e-8)), dim=-1))
            total_loss_mb = actor_loss_mb + 0.5 * critic_loss_mb - 0.01 * entropy_mb

            ppo_optimizer.zero_grad();
            total_loss_mb.backward();
            torch.nn.utils.clip_grad_norm_(policy_to_update.parameters(), 0.5)
            ppo_optimizer.step()

    if use_penalty_lagrangian and cost_flags_batch.numel() > 0:
        avg_cost_in_batch = cost_flags_batch.mean().item()
        updated_lambda_penalty_val = max(0.0, current_lambda_penalty_val + learning_rate_for_lambda * avg_cost_in_batch)
        return updated_lambda_penalty_val

    return current_lambda_penalty_val


env_id = PPO_ENV_ID
PPO_ACTION_DIM = ENVIRONMENT_ACTION_DIM;
STATE_DIM = ENVIRONMENT_STATE_DIM
CONSTRAINT_SAMPLER_ACTION_DIM = PRETRAINED_ACTION_DIM
print(f"--- Sanity Check ---");
print(f"Env Action Dim (PPO Policy Output): {PPO_ACTION_DIM}");
print(f"Pretrained Constraint Encoder Action Dim: {PRETRAINED_ACTION_DIM}");
print(f"Curriculum Sampler / Constraint Definition Action Dim: {CONSTRAINT_SAMPLER_ACTION_DIM}")

# ##############################################################################
# Model and Optimizer Initialization (Decoupled)
# ##############################################################################
PPO_JOINT_EMBED_DIM = 32

# --- Models and Optimizer for Hard Projection Policy ---
embed_net_hard = StateConstraintEmbedder(STATE_DIM, PRETRAINED_D_MODEL, PPO_JOINT_EMBED_DIM).to(PPO_DEVICE)
hard_policy = HardProjPolicy(PPO_JOINT_EMBED_DIM, PPO_ACTION_DIM).to(PPO_DEVICE)
opt_hard = optim.Adam(list(embed_net_hard.parameters()) + list(hard_policy.parameters()), lr=3e-4)

# --- Models and Optimizer for Lagrange Policy (Decoupled) ---
embed_net_lagrange = StateConstraintEmbedder(STATE_DIM, PRETRAINED_D_MODEL, PPO_JOINT_EMBED_DIM).to(PPO_DEVICE)
lagrange_policy = LagrangePolicy(PPO_JOINT_EMBED_DIM, PPO_ACTION_DIM).to(PPO_DEVICE)
opt_lagrange = optim.Adam(list(embed_net_lagrange.parameters()) + list(lagrange_policy.parameters()), lr=3e-4)

# --- Model and Optimizer for Plain PPO Policy ---
ppo_plain = PlainPPOPolicy(STATE_DIM, PPO_ACTION_DIM).to(PPO_DEVICE)
opt_ppo = optim.Adam(list(ppo_plain.parameters()), lr=3e-4)

# --- Shared Operators ---
sdf_proj_operator = SDFProjector(tau=0.05).to(PPO_DEVICE)

lr_lambda = 1e-4
N_SAMPLES_FOR_CONSTRAINT_ENCODING = 30;
SAMPLING_RANGE_FOR_ENCODING = (-2.0, 2.0)
SIGNIFICANT_CORRECTION_THRESHOLD = 0.1

EVAL_FREQ = 50
N_EVAL_EPISODES_PERIODIC = 5


def run_periodic_evaluation(
        current_episode_num: int,
        eval_constraint_raw: Dict[str, Any],
        p_embed_net_hard: StateConstraintEmbedder,
        p_embed_net_lagrange: StateConstraintEmbedder,
        p_hard_policy: HardProjPolicy,
        p_lagrange_policy: LagrangePolicy,
        p_ppo_plain: PlainPPOPolicy,
        p_sdf_proj_operator: SDFProjector,
        device: torch.device,
        num_eval_episodes: int,
        env_id_eval: str,
        n_samples_encoding: int,
        sampling_range_encoding: Tuple[float, float]
):
    eval_constraint_name = eval_constraint_raw['name']
    print(f"--- Running Periodic Evaluation at Ep {current_episode_num} for Constraint: {eval_constraint_name} ---")

    policy_types_to_eval = ['hard', 'lagrange', 'ppo', 'random']
    results_rewards = {ptype: [] for ptype in policy_types_to_eval}
    results_sats = {ptype: [] for ptype in policy_types_to_eval}

    # Encode the constraint only if it's not a PPO/random policy test
    e_constraint_eval_periodic = None
    if eval_constraint_raw.get('type') != 'unconstrained':
        e_constraint_eval_periodic = new_encode_constraint(
            eval_constraint_raw, n_samples_encoding, sampling_range_encoding, device
        ).unsqueeze(0)

    policy_modules_map = {
        'hard': p_hard_policy,
        'lagrange': p_lagrange_policy,
        'ppo': p_ppo_plain,
        'random': None
    }
    if p_embed_net_hard: p_embed_net_hard.eval()
    if p_embed_net_lagrange: p_embed_net_lagrange.eval()
    for p_module in [p_hard_policy, p_lagrange_policy, p_ppo_plain]:
        if p_module: p_module.eval()

    for policy_name in policy_types_to_eval:
        current_policy_module = policy_modules_map.get(policy_name)

        for _ep_eval in range(num_eval_episodes):
            env_eval = gym.make(env_id_eval)
            obs_np, _ = env_eval.reset()
            done_eval, truncated_eval = False, False
            episode_reward_eval, num_steps_eval, num_satisfied_steps_eval = 0.0, 0, 0

            max_episode_length_eval = getattr(env_eval.spec, 'max_episode_steps', 1000)

            for _step in range(max_episode_length_eval):
                if done_eval or truncated_eval: break

                obs_tensor_eval = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                action_from_policy_eval: torch.Tensor
                action_to_execute_eval: torch.Tensor

                constraint_sets_for_eval = eval_constraint_raw.get('sets', [])

                with torch.no_grad():
                    if policy_name == 'random':
                        action_from_policy_eval = torch.from_numpy(env_eval.action_space.sample()).float().to(device)
                        action_to_execute_eval = action_from_policy_eval
                    elif policy_name == 'ppo':
                        mu_eval, _, _ = current_policy_module(obs_tensor_eval)
                        action_from_policy_eval = mu_eval[0] if mu_eval.ndim > 1 else mu_eval
                        action_to_execute_eval = action_from_policy_eval
                    else:  # hard and lagrange
                        joint_embedding_eval = p_embed_net_hard(obs_tensor_eval, e_constraint_eval_periodic)[
                            0] if policy_name == 'hard' else \
                            p_embed_net_lagrange(obs_tensor_eval, e_constraint_eval_periodic)[0]
                        mu_eval, _, _ = current_policy_module(joint_embedding_eval.unsqueeze(0))
                        action_from_policy_eval = mu_eval[0] if mu_eval.ndim > 1 else mu_eval

                        if policy_name == 'hard':
                            if torch.isnan(action_from_policy_eval).any():
                                action_to_execute_eval = torch.nan_to_num(action_from_policy_eval)
                            else:
                                action_after_sdf_eval = action_from_policy_eval
                                if not eval_constraint_raw.get('convex', False):
                                    for _iter_sdf in range(3):
                                        action_after_sdf_eval = p_sdf_proj_operator(action_after_sdf_eval,
                                                                                    constraint_sets_for_eval)

                                if torch.isnan(action_after_sdf_eval).any():
                                    action_to_execute_eval = torch.nan_to_num(action_after_sdf_eval)
                                else:
                                    action_to_execute_eval = hard_project_to_union_of_convex_sets(
                                        action_after_sdf_eval, constraint_sets_for_eval, device
                                    )
                        else:  # Lagrange policy
                            action_to_execute_eval = action_from_policy_eval

                if torch.isnan(action_to_execute_eval).any():
                    action_to_execute_eval = torch.nan_to_num(action_to_execute_eval)

                action_projected_np = action_to_execute_eval.detach().cpu().numpy().flatten()
                action_clamped_np = np.clip(
                    action_projected_np,
                    env_eval.action_space.low,
                    env_eval.action_space.high
                )

                obs_next_np, reward_val, done_eval, truncated_eval, _ = env_eval.step(action_clamped_np)
                episode_reward_eval += reward_val
                num_steps_eval += 1
                obs_np = obs_next_np

                action_to_check_violation_final = action_to_execute_eval.to(device)

                # Check satisfaction for ALL policies against the current constraint
                if not constraint_sets_for_eval:  # If no constraint (e.g., for a PPO-only plot), consider it satisfied
                    num_satisfied_steps_eval += 1
                elif torch.isnan(action_to_check_violation_final).any():
                    pass
                else:
                    is_satisfied_this_step = False
                    for s_set in constraint_sets_for_eval:
                        # Handle both object and dict representations
                        if isinstance(s_set, dict):
                            cfunc = build_cfunc_from_raw_data(s_set, action_to_check_violation_final.device)
                            violation = cfunc(action_to_check_violation_final)
                        else:
                            violation = s_set.violation(action_to_check_violation_final)

                        if violation.item() <= TOL:
                            is_satisfied_this_step = True
                            break  # One satisfied component is enough for a union

                    if is_satisfied_this_step:
                        num_satisfied_steps_eval += 1

            results_rewards[policy_name].append(episode_reward_eval)
            results_sats[policy_name].append(num_satisfied_steps_eval / max(1, num_steps_eval))
            env_eval.close()

    if p_embed_net_hard: p_embed_net_hard.train()
    if p_embed_net_lagrange: p_embed_net_lagrange.train()
    for p_module in [p_hard_policy, p_lagrange_policy, p_ppo_plain]:
        if p_module: p_module.train()

    avg_rewards_eval = {ptype: np.mean(res_list) if res_list else 0.0 for ptype, res_list in results_rewards.items()}
    avg_sats_eval = {ptype: np.mean(res_list) if res_list else 0.0 for ptype, res_list in results_sats.items()}

    print(f"--- Periodic Evaluation Results for {eval_constraint_name} (Ep: {current_episode_num}) ---")
    for p_name_log in policy_types_to_eval:
        print(
            f"  {p_name_log.capitalize():<10}: Avg Reward: {avg_rewards_eval[p_name_log]:.2f}, Avg Sat Rate: {avg_sats_eval[p_name_log]:.3f}")
    print("---------------------------------------------------------------")
    return avg_rewards_eval, avg_sats_eval


def train_loop(episodes=50, batch_steps=128, clip_eps=0.2, ppo_epochs=2, mb_size=16):
    print("--- Initializing Training Loop (Curriculum on Proj Correction, Periodic Eval) ---")
    sampler = CurriculumSampler(CONSTRAINT_SAMPLER_ACTION_DIM)
    env_h, env_p, env_u = gym.make(env_id), gym.make(env_id), gym.make(env_id)

    buf_h, buf_p, buf_u = [], [], [];
    current_lambda_penalty = 1.0

    CURRICULUM_LAMBDA_DIFFICULTY = 1.0
    CURRICULUM_ETA_UPDATE_RATE = 0.1

    max_episode_len_train = getattr(env_h.spec, 'max_episode_steps', 1000)

    # ##########################################################################
    # ### MODIFICATION START: Define the 4 specific test cases for evaluation ###
    # ##########################################################################
    action_dim_for_test_constraints = max(1, PPO_ACTION_DIM)

    # Test Case 1: Benevolent Constraint (Sanity Check)
    test_case_1_benevolent = {
        'name': "Test 1: Benevolent Constraint", 'type': 'l2_norm',
        'params': {'center': np.array([0.0] * action_dim_for_test_constraints), 'radius': 2.0},
        'convex': True,
        'sets': [ConvexBall(np.array([0.0] * action_dim_for_test_constraints), 2.0)],
        'action_dim': PPO_ACTION_DIM
    }

    # Test Case 2: Simple Banishment (Basic Projection)
    test_case_2_banishment = {
        'name': "Test 2: Simple Banishment", 'type': 'l2_norm',
        'params': {'center': np.array([1.5] * action_dim_for_test_constraints), 'radius': 0.2},
        'convex': True,
        'sets': [ConvexBall(np.array([1.5] * action_dim_for_test_constraints), 0.2)],
        'action_dim': PPO_ACTION_DIM
    }

    # Test Case 3: Fork in the Road (Non-Convex Decision)
    c1 = np.array([-0.8] * action_dim_for_test_constraints)  # Better choice
    r1 = 0.3
    c2 = np.array([2.0] * action_dim_for_test_constraints)  # Worse choice
    r2 = 0.3
    test_case_3_fork = {
        'name': "Test 3: Fork in the Road", 'type': 'union',
        'params': {'components': [
            {'type': 'l2_norm', 'params': {'center': c1, 'radius': r1}},
            {'type': 'l2_norm', 'params': {'center': c2, 'radius': r2}}
        ]},
        'convex': False,
        'sets': [ConvexBall(c1, r1), ConvexBall(c2, r2)],
        'action_dim': PPO_ACTION_DIM
    }

    # Test Case 4: Unseen Geometry (Generalization to Box)
    test_case_4_unseen_geom = {
        'name': "Test 4: Unseen Geometry (Box)", 'type': 'box',
        'params': {'low': np.array([0.5] * action_dim_for_test_constraints),
                   'high': np.array([1.5] * action_dim_for_test_constraints)},
        'convex': True,
        # Pass the dict itself in 'sets' so hard_project can handle it
        'sets': [{'type': 'box', 'params': {'low': np.array([0.5] * action_dim_for_test_constraints),
                                            'high': np.array([1.5] * action_dim_for_test_constraints)}}],
        'action_dim': PPO_ACTION_DIM
    }

    evaluation_test_cases = {
        "test1_benevolent": test_case_1_benevolent,
        "test2_banishment": test_case_2_banishment,
        "test3_fork": test_case_3_fork,
        "test4_unseen_geom": test_case_4_unseen_geom,
    }

    eval_log = {
        'episodes': [],
        **{key: {ptype: {'R': [], 'S': []} for ptype in ['hard', 'lagrange', 'ppo', 'random']} for key in
           evaluation_test_cases.keys()}
    }
    # ##########################################################################
    # ### MODIFICATION END ###
    # ##########################################################################

    for ep in range(episodes):
        current_training_constraint_raw = sampler.sample()

        obs_h_np, _ = env_h.reset(seed=ep);
        obs_p_np, _ = env_p.reset(seed=ep);
        obs_u_np, _ = env_u.reset(seed=ep);

        e_constraint_for_training = new_encode_constraint(
            current_training_constraint_raw, N_SAMPLES_FOR_CONSTRAINT_ENCODING,
            SAMPLING_RANGE_FOR_ENCODING, PPO_DEVICE
        ).unsqueeze(0)

        current_joint_embedding_h = \
            embed_net_hard(torch.tensor(obs_h_np, dtype=torch.float32, device=PPO_DEVICE),
                           e_constraint_for_training.squeeze(0))[
                0].detach()
        current_joint_embedding_p = \
            embed_net_lagrange(torch.tensor(obs_p_np, dtype=torch.float32, device=PPO_DEVICE),
                               e_constraint_for_training.squeeze(0))[
                0].detach()
        current_state_u_tensor = torch.tensor(obs_u_np, dtype=torch.float32, device=PPO_DEVICE)

        ep_buffer_h, ep_buffer_p, ep_buffer_u = [], [], []
        ep_reward_h, ep_reward_p, ep_reward_u = 0.0, 0.0, 0.0
        ep_satisfied_steps_h, ep_satisfied_steps_p, ep_satisfied_steps_u = 0, 0, 0
        ep_num_steps_h, ep_num_steps_p, ep_num_steps_u = 0, 0, 0
        done_h, done_p, done_u = False, False, False

        episode_accumulated_proj_correction_metric_h = 0.0

        for _step_train in range(max_episode_len_train):
            if done_h and done_p and done_u: break

            if not done_h:
                mean_h, std_h, val_h = hard_policy(current_joint_embedding_h);
                action_policy_raw_h = mean_h + std_h * torch.randn_like(mean_h)

                action_for_learning_h = action_policy_raw_h
                action_to_execute_h: torch.Tensor
                action_to_measure_correction_from_h = action_policy_raw_h

                if torch.isnan(action_policy_raw_h).any():
                    action_policy_raw_h = torch.nan_to_num(action_policy_raw_h)
                    action_to_measure_correction_from_h = action_policy_raw_h

                action_after_sdf_h = action_policy_raw_h
                if not current_training_constraint_raw['convex']:
                    action_after_sdf_h = sdf_proj_operator(action_policy_raw_h,
                                                           current_training_constraint_raw['sets'])

                if torch.isnan(action_after_sdf_h).any():
                    action_after_sdf_h = torch.nan_to_num(action_after_sdf_h)

                action_to_measure_correction_from_h = action_after_sdf_h
                action_to_execute_h = hard_project_to_union_of_convex_sets(
                    action_after_sdf_h, current_training_constraint_raw['sets'], PPO_DEVICE
                )

                if torch.isnan(action_to_execute_h).any():
                    action_to_execute_h = torch.nan_to_num(action_to_execute_h)

                projection_correction_dist_h = torch.linalg.norm(
                    action_to_measure_correction_from_h - action_to_execute_h)
                if projection_correction_dist_h.item() > SIGNIFICANT_CORRECTION_THRESHOLD:
                    episode_accumulated_proj_correction_metric_h += 1.0

                log_prob_old_h = gaussian_logp(action_for_learning_h, mean_h, std_h).detach()

                constraint_sets_h = current_training_constraint_raw.get('sets', [])
                is_satisfied_h_step = True
                if not torch.isnan(action_to_execute_h).any() and constraint_sets_h:
                    violations_h_list = [s.violation(action_to_execute_h) for s in constraint_sets_h if
                                         hasattr(s, 'violation')]
                    if violations_h_list:
                        is_satisfied_h_step = bool(torch.min(torch.stack(violations_h_list)).item() <= TOL)
                elif torch.isnan(action_to_execute_h).any():
                    is_satisfied_h_step = False

                cost_flag_h = not is_satisfied_h_step

                ep_buffer_h.append((current_joint_embedding_h.detach(), action_for_learning_h.detach(), log_prob_old_h,
                                    0.0, val_h.item(), cost_flag_h))
                obs_next_h_np, reward_h_step, terminated_h, truncated_h, _ = env_h.step(
                    action_to_execute_h.detach().cpu().numpy())
                done_h = terminated_h or truncated_h;
                ep_reward_h += reward_h_step;
                ep_num_steps_h += 1;
                ep_satisfied_steps_h += int(is_satisfied_h_step);
                ep_buffer_h[-1] = (*ep_buffer_h[-1][:3], reward_h_step, *ep_buffer_h[-1][4:])

                if not done_h:
                    current_joint_embedding_h = \
                        embed_net_hard(torch.tensor(obs_next_h_np, dtype=torch.float32, device=PPO_DEVICE),
                                       e_constraint_for_training.squeeze(0))[0].detach()

            if not done_p:
                mean_p, std_p, val_p = lagrange_policy(current_joint_embedding_p);
                action_to_execute_p = mean_p + std_p * torch.randn_like(mean_p);
                if torch.isnan(action_to_execute_p).any():
                    action_to_execute_p = torch.nan_to_num(action_to_execute_p)

                log_prob_old_p = gaussian_logp(action_to_execute_p, mean_p, std_p).detach()

                constraint_sets_p = current_training_constraint_raw.get('sets', [])
                is_satisfied_p_step = True
                if not torch.isnan(action_to_execute_p).any() and constraint_sets_p:
                    violations_p_list = [s.violation(action_to_execute_p) for s in constraint_sets_p if
                                         hasattr(s, 'violation')]
                    if violations_p_list:
                        is_satisfied_p_step = bool(torch.min(torch.stack(violations_p_list)).item() <= TOL)
                elif torch.isnan(action_to_execute_p).any():
                    is_satisfied_p_step = False
                cost_flag_p = not is_satisfied_p_step

                ep_buffer_p.append((current_joint_embedding_p.detach(), action_to_execute_p.detach(), log_prob_old_p,
                                    0.0, val_p.item(), cost_flag_p))
                obs_next_p_np, reward_p_step, terminated_p, truncated_p, _ = env_p.step(
                    action_to_execute_p.detach().cpu().numpy())
                done_p = terminated_p or truncated_p;
                ep_reward_p += reward_p_step;
                ep_num_steps_p += 1;
                ep_satisfied_steps_p += int(is_satisfied_p_step);
                ep_buffer_p[-1] = (*ep_buffer_p[-1][:3], reward_p_step, *ep_buffer_p[-1][4:])
                if not done_p:
                    current_joint_embedding_p = \
                        embed_net_lagrange(torch.tensor(obs_next_p_np, dtype=torch.float32, device=PPO_DEVICE),
                                           e_constraint_for_training.squeeze(0))[0].detach()

            if not done_u:
                mean_u, std_u, val_u = ppo_plain(current_state_u_tensor);
                action_to_execute_u = mean_u + std_u * torch.randn_like(mean_u);
                if torch.isnan(action_to_execute_u).any():
                    action_to_execute_u = torch.nan_to_num(action_to_execute_u)

                log_prob_old_u = gaussian_logp(action_to_execute_u, mean_u, std_u).detach()

                constraint_sets_u = current_training_constraint_raw.get('sets', [])
                is_satisfied_u_step = True
                if not torch.isnan(action_to_execute_u).any() and constraint_sets_u:
                    violations_u_list = [s.violation(action_to_execute_u) for s in constraint_sets_u if
                                         hasattr(s, 'violation')]
                    if violations_u_list:
                        is_satisfied_u_step = bool(torch.min(torch.stack(violations_u_list)).item() <= TOL)
                elif torch.isnan(action_to_execute_u).any():
                    is_satisfied_u_step = False
                cost_flag_u = not is_satisfied_u_step

                ep_buffer_u.append((current_state_u_tensor.detach(), action_to_execute_u.detach(), log_prob_old_u, 0.0,
                                    val_u.item(), cost_flag_u))
                obs_next_u_np, reward_u_step, terminated_u, truncated_u, _ = env_u.step(
                    action_to_execute_u.detach().cpu().numpy())
                done_u = terminated_u or truncated_u;
                ep_reward_u += reward_u_step;
                ep_num_steps_u += 1;
                ep_satisfied_steps_u += int(is_satisfied_u_step);
                ep_buffer_u[-1] = (*ep_buffer_u[-1][:3], reward_u_step, *ep_buffer_u[-1][4:])
                if not done_u:
                    current_state_u_tensor = torch.tensor(obs_next_u_np, dtype=torch.float32, device=PPO_DEVICE)

        if ep_buffer_h: finish_episode(ep_buffer_h); buf_h.extend(ep_buffer_h)
        if ep_buffer_p: finish_episode(ep_buffer_p); buf_p.extend(ep_buffer_p)
        if ep_buffer_u: finish_episode(ep_buffer_u); buf_u.extend(ep_buffer_u)

        avg_ep_proj_correction_h = episode_accumulated_proj_correction_metric_h / max(1,
                                                                                      ep_num_steps_h) if ep_num_steps_h > 0 else 0
        sampler.record_difficulty_metric(current_training_constraint_raw['level'], avg_ep_proj_correction_h)

        if len(buf_h) >= batch_steps:
            ppo_update(buf_h, hard_policy, opt_hard, clip_eps, ppo_epochs, mb_size, is_plain_ppo_policy=False,
                       use_penalty_lagrangian=False);
            buf_h.clear()
        if len(buf_p) >= batch_steps:
            current_lambda_penalty = ppo_update(
                buf_p, lagrange_policy, opt_lagrange, clip_eps, ppo_epochs, mb_size,
                is_plain_ppo_policy=False, use_penalty_lagrangian=True,
                current_lambda_penalty_val=current_lambda_penalty, learning_rate_for_lambda=lr_lambda
            );
            buf_p.clear()
        if len(buf_u) >= batch_steps:
            ppo_update(buf_u, ppo_plain, opt_ppo, clip_eps, ppo_epochs, mb_size, is_plain_ppo_policy=True,
                       use_penalty_lagrangian=False);
            buf_u.clear()

        if (ep + 1) % 10 == 0 or ep == episodes - 1:
            sat_rate_h = ep_satisfied_steps_h / max(1, ep_num_steps_h) if ep_num_steps_h > 0 else 0
            sat_rate_p = ep_satisfied_steps_p / max(1, ep_num_steps_p) if ep_num_steps_p > 0 else 0
            sat_rate_u = ep_satisfied_steps_u / max(1, ep_num_steps_u) if ep_num_steps_u > 0 else 0
            print(
                f"Ep:{ep + 1:4d}, Lvl:{current_training_constraint_raw['level']}, R_h:{ep_reward_h:6.1f},S_h:{sat_rate_h:.2f} | "
                f"R_p:{ep_reward_p:6.1f},S_p:{sat_rate_p:.2f},lam:{current_lambda_penalty:5.2f} | "
                f"R_u:{ep_reward_u:6.1f},S_u:{sat_rate_u:.2f}")
            sampler.update_weights(eta=CURRICULUM_ETA_UPDATE_RATE, difficulty_lambda=CURRICULUM_LAMBDA_DIFFICULTY)

        # ##########################################################################
        # ### MODIFICATION START: Iterate over all test cases for evaluation ###
        # ##########################################################################
        if (ep + 1) % EVAL_FREQ == 0 or ep == episodes - 1:
            eval_log['episodes'].append(ep + 1)

            for test_key, test_constraint_raw in evaluation_test_cases.items():
                avg_rewards, avg_sats = run_periodic_evaluation(
                    current_episode_num=ep + 1,
                    eval_constraint_raw=test_constraint_raw,
                    p_embed_net_hard=embed_net_hard,
                    p_embed_net_lagrange=embed_net_lagrange,
                    p_hard_policy=hard_policy,
                    p_lagrange_policy=lagrange_policy,
                    p_ppo_plain=ppo_plain,
                    p_sdf_proj_operator=sdf_proj_operator,
                    device=PPO_DEVICE,
                    num_eval_episodes=N_EVAL_EPISODES_PERIODIC,
                    env_id_eval=env_id,
                    n_samples_encoding=N_SAMPLES_FOR_CONSTRAINT_ENCODING,
                    sampling_range_encoding=SAMPLING_RANGE_FOR_ENCODING
                )
                for p_key_eval in ['hard', 'lagrange', 'ppo', 'random']:
                    eval_log[test_key][p_key_eval]['R'].append(avg_rewards[p_key_eval])
                    eval_log[test_key][p_key_eval]['S'].append(avg_sats[p_key_eval])
        # ##########################################################################
        # ### MODIFICATION END ###
        # ##########################################################################

    env_h.close();
    env_p.close();
    env_u.close();
    # ##########################################################################
    # ### MODIFICATION START: Return test case definitions for plotting ###
    # ##########################################################################
    return eval_log, evaluation_test_cases
    # ##########################################################################
    # ### MODIFICATION END ###
    # ##########################################################################


def main():
    print(f"--- Starting PPO Main Script ---")
    num_training_episodes = 5000 # Can be adjusted
    training_batch_steps = 256;
    ppo_mini_batch_size = 32;
    num_ppo_update_epochs = 4;

    print(f"Running training for {num_training_episodes} episodes...")
    training_start_time = time.time();

    # ##########################################################################
    # ### MODIFICATION START: Receive test case definitions from train_loop ###
    # ##########################################################################
    evaluation_log_results, evaluation_test_cases = train_loop(
        episodes=num_training_episodes, batch_steps=training_batch_steps,
        mb_size=ppo_mini_batch_size, ppo_epochs=num_ppo_update_epochs
    )
    # ##########################################################################
    # ### MODIFICATION END ###
    # ##########################################################################

    print(f"Training loop finished in {(time.time() - training_start_time):.2f} seconds.")

    # ##########################################################################
    # ### MODIFICATION START: Plot each test case in a separate figure ###
    # ##########################################################################
    if not evaluation_log_results or not evaluation_log_results['episodes']:
        print("No evaluation data collected during training. Skipping plots.")
    else:
        evaluation_episodes_axis = evaluation_log_results['episodes']

        # Define policies to plot and their styles
        plot_policy_keys = ['hard', 'lagrange', 'ppo', 'random']
        plot_policy_labels = {'hard': 'Hard+Proj', 'lagrange': 'Lagrange', 'ppo': 'PPO (Unconstrained)',
                              'random': 'Random'}
        plot_policy_markers = {'hard': 'o', 'lagrange': 'x', 'ppo': 's', 'random': '^'}
        plot_policy_linestyles = {'hard': '-', 'lagrange': '--', 'ppo': ':', 'random': '-.'}

        # Get the keys for the test cases used in the log
        test_case_keys = [key for key in evaluation_log_results if key != 'episodes']

        # Loop through each test case and create a separate figure
        for test_key in test_case_keys:
            # Extract the readable name from the constraint dict using the returned definitions
            test_case_name = evaluation_test_cases[test_key].get('name', test_key)

            # Create a new figure with two subplots (1 row, 2 columns)
            fig, (ax_reward, ax_sat) = plt.subplots(1, 2, figsize=(18, 6))
            fig.suptitle(f'Periodic Evaluation for: {test_case_name}', fontsize=16)

            # --- Plot Rewards on the left subplot ---
            for p_key in plot_policy_keys:
                if p_key in evaluation_log_results[test_key] and evaluation_log_results[test_key][p_key]['R']:
                    ax_reward.plot(evaluation_episodes_axis, evaluation_log_results[test_key][p_key]['R'],
                                   label=plot_policy_labels[p_key], marker=plot_policy_markers[p_key],
                                   linestyle=plot_policy_linestyles[p_key])
            ax_reward.set_title("Reward Progression")
            ax_reward.set_xlabel("Training Episodes")
            ax_reward.set_ylabel("Log Reward")
            ax_reward.legend()
            ax_reward.grid(True)
            ax_reward.set_yscale('log')  # Set Y-axis to log scale

            # --- Plot Satisfaction Rates on the right subplot ---
            for p_key in plot_policy_keys:
                if p_key in evaluation_log_results[test_key] and evaluation_log_results[test_key][p_key]['S']:
                    ax_sat.plot(evaluation_episodes_axis, evaluation_log_results[test_key][p_key]['S'],
                                label=plot_policy_labels[p_key], marker=plot_policy_markers[p_key],
                                linestyle=plot_policy_linestyles[p_key])
            ax_sat.set_title("Satisfaction Rate Progression")
            ax_sat.set_xlabel("Training Episodes")
            ax_sat.set_ylabel("Avg Sat Rate")
            ax_sat.legend()
            ax_sat.grid(True)
            ax_sat.set_ylim(-0.05, 1.05)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # Show the current figure. The script will pause here until you close the plot window.
            plt.show()

    # ##########################################################################
    # ### MODIFICATION END ###
    # ##########################################################################


if __name__ == '__main__':
    print(f"--- PPO Script Started ---")
    # Sanity checks from original code...
    error_messages_main = []
    if CONSTRAINT_SAMPLER_ACTION_DIM != PRETRAINED_ACTION_DIM:
        error_messages_main.append(
            "CRITICAL WARNING: Mismatch: CONSTRAINT_SAMPLER_ACTION_DIM != PRETRAINED_ACTION_DIM!")
    if PPO_ACTION_DIM != CONSTRAINT_SAMPLER_ACTION_DIM:
        error_messages_main.append(
            f"CRITICAL WARNING: Mismatch: PPO_ACTION_DIM ({PPO_ACTION_DIM}) != CONSTRAINT_SAMPLER_ACTION_DIM ({CONSTRAINT_SAMPLER_ACTION_DIM})!")
    if PPO_ACTION_DIM == 0:
        error_messages_main.append(f"CRITICAL WARNING: PPO_ACTION_DIM is 0.")

    if error_messages_main:
        for msg in error_messages_main: print(msg)
        print("Please resolve critical dimension mismatches or issues before proceeding. Exiting.")
        exit()

    main()
    print(f"--- PPO Script Finished ---")