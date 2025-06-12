#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embedding_hopper_gpu.py (Adapted from 1D version for Hopper-v4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Callable, Dict, Any, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import time  # To measure training time


# --- 0. NT-Xent Loss ---
class NT_Xent_Loss(nn.Module):
    def __init__(self, batch_size_effective: int, temperature: float, device: torch.device):
        super(NT_Xent_Loss, self).__init__()
        self.batch_size_effective = batch_size_effective
        self.temperature = temperature
        self.device = device

        num_original_constraints = batch_size_effective // 2
        labels = torch.arange(batch_size_effective, device=self.device)
        # Positive pairs are (z_i, z_{i+N}) and (z_{i+N}, z_i)
        self.positive_pair_indices = torch.cat([
            labels[num_original_constraints:],
            labels[:num_original_constraints]
        ])
        # Mask to exclude self-similarity
        self.identity_mask = (~torch.eye(batch_size_effective, dtype=torch.bool, device=self.device))

    def forward(self, z_all: torch.Tensor) -> torch.Tensor:
        z_all_norm = F.normalize(z_all, dim=1)
        sim_matrix = torch.mm(z_all_norm, z_all_norm.t())
        sim_matrix_scaled = sim_matrix / self.temperature

        # Get the similarity of positive pairs
        sim_positive = sim_matrix_scaled[torch.arange(z_all.shape[0], device=self.device), self.positive_pair_indices]

        # Numerator of the loss function
        numerator = torch.exp(sim_positive)

        # Denominator of the loss function (sum of similarities with all other samples in the batch)
        exp_sim_negatives = torch.exp(sim_matrix_scaled) * self.identity_mask
        denominator = torch.sum(exp_sim_negatives, dim=1)

        # Calculate the loss
        loss = -torch.log(numerator / (denominator + 1e-8))
        return loss.mean()


# --- 1. ConstraintEmbeddingTransformer (Base Encoder f(·)) ---
class ConstraintEmbeddingTransformer(nn.Module):
    def __init__(self, action_dim: int, d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        self.action_dim = action_dim
        self.d_model = d_model
        self.input_token_dim = action_dim + 1
        self.linear_projection = nn.Linear(self.input_token_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation=F.gelu
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, constraint_tokens_batch: torch.Tensor) -> torch.Tensor:
        if constraint_tokens_batch.shape[-1] != self.input_token_dim:
            raise ValueError(f"Input token dimension mismatch. Expected {self.input_token_dim}, "
                             f"got {constraint_tokens_batch.shape[-1]} (model action_dim={self.action_dim})")
        batch_size = constraint_tokens_batch.shape[0]
        projected_tokens = self.linear_projection(constraint_tokens_batch)
        S = torch.cat([self.cls_token.expand(batch_size, -1, -1), projected_tokens], dim=1)
        transformer_output = self.transformer_encoder(S)
        cls_output_h_prime = transformer_output[:, 0, :]
        return cls_output_h_prime


# --- 2. SimCLR Model Wrapper (Base Encoder + Projection Head g(·)) ---
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder: ConstraintEmbeddingTransformer, projection_dim: int = 128):
        super().__init__()
        self.base_encoder = base_encoder
        h_prime_dim = base_encoder.d_model
        self.projection_head = nn.Sequential(
            nn.Linear(h_prime_dim, h_prime_dim),
            nn.ReLU(),
            nn.Linear(h_prime_dim, projection_dim)
        )

    def forward(self, constraint_tokens_batch: torch.Tensor) -> torch.Tensor:
        h_prime = self.base_encoder(constraint_tokens_batch)
        z = self.projection_head(h_prime)
        return z


# --- 3. `cfunc` Implementations & Dispatcher ---
ConstraintFunction = Callable[[torch.Tensor], torch.Tensor]


def cfunc_l2_norm(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    return torch.linalg.norm(x - center.to(x.device), ord=2, dim=-1) - radius


def cfunc_l1_norm(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    return torch.linalg.norm(x - center.to(x.device), ord=1, dim=-1) - radius


def cfunc_linear_halfspace(x: torch.Tensor, a: torch.Tensor, b: float) -> torch.Tensor:
    a_dev = a.to(x.device)
    if x.ndim >= 1 and a_dev.ndim == 1 and x.shape[-1] == a_dev.shape[0]:
        return torch.sum(x * a_dev, dim=-1) - b
    raise ValueError(f"Shape mismatch in cfunc_linear_halfspace: x={x.shape}, a={a_dev.shape}")


def cfunc_box(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    low_dev, high_dev = low.to(x.device), high.to(x.device)
    violations_high = x - high_dev
    violations_low = low_dev - x
    all_violations = torch.cat([violations_high, violations_low], dim=-1)
    return torch.max(all_violations, dim=-1).values


def cfunc_polytope(x: torch.Tensor, A_matrix: torch.Tensor, b_vector: torch.Tensor) -> torch.Tensor:
    A_dev, b_dev = A_matrix.to(x.device), b_vector.to(x.device)
    violations = torch.matmul(x, A_dev.transpose(-1, -2)) - b_dev
    return torch.max(violations, dim=-1).values


def cfunc_ellipsoid(x: torch.Tensor, center: torch.Tensor, P_inv: torch.Tensor) -> torch.Tensor:
    center_dev, P_inv_dev = center.to(x.device), P_inv.to(x.device)
    diff = x - center_dev
    if diff.ndim == 1 and P_inv_dev.ndim == 1:  # 1D case
        return P_inv_dev.squeeze() * diff * diff - 1.0
    elif diff.ndim > 1 and P_inv_dev.ndim == 1:  # Batched 1D case
        return P_inv_dev.squeeze() * torch.sum(diff * diff, dim=-1) - 1.0
    # N-D case
    return torch.sum(torch.matmul(diff.unsqueeze(-2), P_inv_dev).squeeze(-2) * diff, dim=-1) - 1.0


def cfunc_soc(x: torch.Tensor, A_soc: torch.Tensor, b_soc: torch.Tensor, c_soc: torch.Tensor,
              d_soc: float) -> torch.Tensor:
    A_dev, b_dev, c_dev = A_soc.to(x.device), b_soc.to(x.device), c_soc.to(x.device)
    is_batched = x.ndim > 1

    if A_dev.numel() > 0:
        Ax_plus_b = torch.matmul(x, A_dev.t()) + b_dev if is_batched else A_dev @ x + b_dev
    else:
        Ax_plus_b = b_dev.expand(x.shape[0], -1) if is_batched and b_dev.numel() > 0 else (
            b_dev if b_dev.numel() > 0 else torch.zeros((x.shape[0] if is_batched else 0, 0), device=x.device))

    if c_dev.numel() > 0:
        cTx_plus_d = torch.matmul(x, c_dev) + d_soc if is_batched else c_dev @ x + d_soc
    else:
        cTx_plus_d = torch.full(x.shape[:-1] if is_batched else (), d_soc, device=x.device)

    if Ax_plus_b.shape[-1] == 0:
        norm_Ax_plus_b = torch.zeros_like(cTx_plus_d)
    else:
        norm_Ax_plus_b = torch.linalg.norm(Ax_plus_b, ord=2, dim=-1)

    return norm_Ax_plus_b - cTx_plus_d


class CallableOracleWrapper:
    def __init__(self, oracle: Callable[[torch.Tensor], Any], device: torch.device):
        self.oracle = oracle
        self.device = device

    def __call__(self, x_i: torch.Tensor) -> torch.Tensor:
        val = self.oracle(x_i.to(self.device))
        if not isinstance(val, torch.Tensor):
            val = torch.tensor([val] if not isinstance(val, (list, tuple, np.ndarray)) else val, dtype=torch.float32,
                               device=self.device)
        if val.ndim == 0:
            val = val.unsqueeze(0)
        return val.float()


def build_cfunc_from_raw_data(raw_data: Dict[str, Any], device: torch.device) -> ConstraintFunction:
    ctype = raw_data.get('type', 'unknown')
    params = raw_data.get('params', {})
    processed_params = {}
    for key, val in params.items():
        if key == 'components' and ctype == 'union':
            processed_params[key] = val
        elif isinstance(val, (np.ndarray, list)):
            try:
                val_np = np.array(val, dtype=np.float32)
            except Exception:
                val_np = np.array(val, dtype=object).astype(np.float32)
            processed_params[key] = torch.from_numpy(val_np).to(device)
        elif isinstance(val, torch.Tensor):
            processed_params[key] = val.to(dtype=torch.float32, device=device)
        else:
            processed_params[key] = val

    if ctype == 'l2_norm':
        return lambda x: cfunc_l2_norm(x, processed_params['center'], processed_params['radius'])
    elif ctype == 'l1_norm':
        return lambda x: cfunc_l1_norm(x, processed_params['center'], processed_params['radius'])
    elif ctype == 'linear_halfspace':
        return lambda x: cfunc_linear_halfspace(x, processed_params['a'], processed_params['b'])
    elif ctype == 'box':
        return lambda x: cfunc_box(x, processed_params['low'], processed_params['high'])
    elif ctype == 'polytope':
        return lambda x: cfunc_polytope(x, processed_params['A'], processed_params['b'])
    elif ctype == 'ellipsoid':
        return lambda x: cfunc_ellipsoid(x, processed_params['center'], processed_params['P_inv'])
    elif ctype == 'soc':
        return lambda x: cfunc_soc(x, processed_params['A_soc'], processed_params['b_soc'], processed_params['c_soc'],
                                   processed_params['d_soc'])
    elif ctype == 'union':
        component_cfuncs = [build_cfunc_from_raw_data(comp, device) for comp in processed_params['components']]

        def cfunc_union(x: torch.Tensor) -> torch.Tensor:
            is_batched = x.ndim > 1
            evals = torch.stack([cf(x) for cf in component_cfuncs], dim=0)
            min_violations, _ = torch.min(evals, dim=0)
            return min_violations

        return cfunc_union
    elif callable(raw_data.get('cfunc_oracle')):
        return CallableOracleWrapper(raw_data['cfunc_oracle'], device)
    else:
        raise ValueError(f"Unsupported ctype: {ctype} in {raw_data}")


# --- 4. Constraint Dataset for SimCLR ---
class ConstraintDatasetSimCLR(Dataset):
    def __init__(self, list_of_constraint_raw_data: List[Dict[str, Any]],
                 N_samples_per_view: int, action_dim: int,
                 sampling_range: Tuple[float, float] = (-2.0, 2.0),
                 device_for_cfunc: torch.device = torch.device("cpu")):
        self.constraint_raw_data_list = list_of_constraint_raw_data
        self.N_samples = N_samples_per_view
        self.action_dim = action_dim
        self.sampling_min, self.sampling_max = sampling_range
        self.sampling_delta = self.sampling_max - self.sampling_min
        self.device_for_cfunc = device_for_cfunc

    def __len__(self):
        return len(self.constraint_raw_data_list)

    def _generate_one_view(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        cfunc = build_cfunc_from_raw_data(raw_data, self.device_for_cfunc)
        tokens: List[torch.Tensor] = []
        for _ in range(self.N_samples):
            x_i = (torch.rand(self.action_dim, device=self.device_for_cfunc) * self.sampling_delta) + self.sampling_min
            c_i_val = cfunc(x_i)
            if not isinstance(c_i_val, torch.Tensor):
                c_i_val = torch.tensor([c_i_val], dtype=torch.float32, device=self.device_for_cfunc)
            if c_i_val.numel() != 1: c_i_val = c_i_val.flatten()[0]
            c_i = c_i_val.reshape(1).float()
            tokens.append(torch.cat([x_i, c_i]))
        return torch.stack(tokens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_data = self.constraint_raw_data_list[idx]
        view1_tokens = self._generate_one_view(raw_data)
        view2_tokens = self._generate_one_view(raw_data)
        return view1_tokens, view2_tokens


# --- 5. Helper: Generate Training Data (Dummy) - ADAPTED FOR N-D ---
def generate_training_raw_data(num_total: int, action_dim: int) -> List[Dict[str, Any]]:
    print(f"Generating training data for ACTION_DIM = {action_dim}")
    all_raw_data = []

    # Define possible types based on action dimension
    base_types = ['l2_norm', 'box', 'linear_halfspace']
    if action_dim == 1:
        # For 1D, some types are degenerate or less meaningful
        possible_types = base_types + ['union']
    else:  # For N-D (action_dim > 1)
        possible_types = base_types + ['polytope', 'ellipsoid', 'soc', 'union']

    for i in range(num_total):
        chosen_type = np.random.choice(possible_types)
        constraint_raw = {'name': f"{chosen_type}_{i}", 'type': chosen_type, 'action_dim': action_dim}
        params = {}

        if chosen_type == 'l2_norm':
            params = {'center': np.random.uniform(-1, 1, size=action_dim), 'radius': np.random.uniform(0.3, 1.0)}
        elif chosen_type == 'box':
            low = np.random.uniform(-1.5, 0, size=action_dim)
            high = low + np.random.uniform(0.2, 1.0, size=action_dim)
            params = {'low': low, 'high': high}
        elif chosen_type == 'linear_halfspace':
            a_vec = np.random.randn(action_dim)
            if np.linalg.norm(a_vec) > 1e-6: a_vec /= np.linalg.norm(a_vec)
            params = {'a': a_vec, 'b': np.random.randn() * 0.5}
        elif chosen_type == 'polytope':  # N-D only
            num_halfspaces = np.random.randint(action_dim + 1, action_dim + 4)
            A = np.random.randn(num_halfspaces, action_dim)
            b = np.random.rand(num_halfspaces) * 0.5 + 0.1
            params = {'A': A, 'b': b}
        elif chosen_type == 'ellipsoid':  # N-D only
            center = np.random.uniform(-1, 1, size=action_dim)
            L = np.random.rand(action_dim, action_dim) * 0.5
            for d_idx in range(action_dim): L[d_idx, d_idx] = np.random.uniform(0.5, 1.5)
            L[np.triu_indices(action_dim, k=1)] = 0
            P_inv = L @ L.T + np.eye(action_dim) * 1e-3
            params = {'center': center, 'P_inv': P_inv}
        elif chosen_type == 'soc':  # N-D only
            m_soc = max(1, action_dim)
            A_s = np.random.randn(m_soc, action_dim) * 0.5
            b_s = np.random.randn(m_soc) * 0.5
            c_s = np.random.randn(action_dim) * 0.1
            d_s = np.random.uniform(np.linalg.norm(b_s) + 0.1, np.linalg.norm(b_s) + 1.0)
            params = {'A_soc': A_s, 'b_soc': b_s, 'c_soc': c_s, 'd_soc': d_s}
        elif chosen_type == 'union':
            num_components = 2
            components = []
            for _ in range(num_components):
                # For union, we stick to simpler components
                comp_type_for_union = np.random.choice(['l2_norm', 'box'])
                comp_raw_child = {'name': f"{comp_type_for_union}_comp", 'type': comp_type_for_union,
                                  'action_dim': action_dim}
                child_params = {}
                if comp_type_for_union == 'l2_norm':
                    child_params = {'center': np.random.uniform(-1.5, 1.5, size=action_dim),
                                    'radius': np.random.uniform(0.1, 0.4)}
                else:  # Box
                    low = np.random.uniform(-1.5, 1, size=action_dim)
                    high = low + np.random.uniform(0.1, 0.4, size=action_dim)
                    child_params = {'low': low, 'high': high}
                comp_raw_child['params'] = child_params
                components.append(comp_raw_child)
            params = {'components': components}

        if params:
            constraint_raw['params'] = params
            all_raw_data.append(constraint_raw)

    return all_raw_data


# --- 6. Helper: Get Embeddings for Inference/Testing ---
def get_embeddings_for_inference(
        base_model: ConstraintEmbeddingTransformer, list_of_raw_data: List[Dict[str, Any]],
        N_samples: int, action_dim_for_tokens: int,
        sampling_range: Tuple[float, float], device: torch.device
) -> torch.Tensor:
    base_model.eval()
    base_model.to(device)
    batch_tokens_list = []

    # Use a temporary dataset to generate views without a DataLoader
    temp_dataset = ConstraintDatasetSimCLR(
        list_of_constraint_raw_data=list_of_raw_data,
        N_samples_per_view=N_samples, action_dim=action_dim_for_tokens,
        sampling_range=sampling_range, device_for_cfunc=device
    )

    for i in range(len(list_of_raw_data)):
        raw_data_item = list_of_raw_data[i]
        view_tokens = temp_dataset._generate_one_view(raw_data_item)
        batch_tokens_list.append(view_tokens)

    if not batch_tokens_list:
        return torch.empty(0, base_model.d_model, device=device)

    batch_tokens_tensor = torch.stack(batch_tokens_list).to(device)
    with torch.no_grad():
        h_prime_embeddings = base_model(batch_tokens_tensor)

    return h_prime_embeddings


# --- 7. Main Pre-training and Testing Script ---
def main_pretrain_and_test():
    print("--- Script Starting: main_pretrain_and_test (Adapted for Hopper-v4) ---")

    # --- MODIFIED FOR HOPPER: Parameters for Hopper-v4 Environment ---
    try:
        import gymnasium as gym
        temp_env = gym.make('Hopper-v4')
        ACTION_DIM = temp_env.action_space.shape[0]
        temp_env.close()
    except Exception as e:
        print(f"Could not create Hopper-v4 env. Defaulting ACTION_DIM to 3. Error: {e}")
        ACTION_DIM = 3

    # MODIFIED FOR HOPPER: Increased model capacity and training scale
    D_MODEL = 128
    N_HEAD = 8
    NUM_ENCODER_LAYERS = 4
    PROJECTION_DIM = 128
    TEMPERATURE = 0.2

    N_SAMPLES_PER_VIEW = 50
    # Hopper action space is [-1, 1]. Sampling range is kept wider.
    SAMPLING_RANGE = (-2.0, 2.0)

    NUM_EPOCHS_PRETRAIN = 100
    BATCH_SIZE_CONSTRAINTS = 64
    LEARNING_RATE = 3e-4  # Standard for Adam
    WEIGHT_DECAY = 1e-5

    # --- Device Configuration ---
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    NUM_WORKERS_DATALOADER = 4 if DEVICE.type == 'cuda' else 0
    PIN_MEMORY_DATALOADER = True if DEVICE.type == 'cuda' else False

    print(f"Running on device: {DEVICE} with ACTION_DIM = {ACTION_DIM}")

    num_training_constraints = BATCH_SIZE_CONSTRAINTS * 20
    print(f"Generating {num_training_constraints} constraints for pre-training...")
    training_constraint_definitions = generate_training_raw_data(num_training_constraints, ACTION_DIM)

    if not training_constraint_definitions:
        print("No training data generated. Aborting.");
        return

    train_dataset = ConstraintDatasetSimCLR(
        list_of_constraint_raw_data=training_constraint_definitions,
        N_samples_per_view=N_SAMPLES_PER_VIEW, action_dim=ACTION_DIM,
        sampling_range=SAMPLING_RANGE, device_for_cfunc=DEVICE
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_CONSTRAINTS, shuffle=True,
        num_workers=NUM_WORKERS_DATALOADER,
        drop_last=True,
        pin_memory=PIN_MEMORY_DATALOADER
    )

    base_encoder = ConstraintEmbeddingTransformer(
        action_dim=ACTION_DIM, d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=D_MODEL * 2, dropout=0.1
    )
    simclr_model = SimCLRModel(base_encoder, projection_dim=PROJECTION_DIM).to(DEVICE)
    optimizer = optim.AdamW(simclr_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * NUM_EPOCHS_PRETRAIN)

    criterion = NT_Xent_Loss(batch_size_effective=BATCH_SIZE_CONSTRAINTS * 2, temperature=TEMPERATURE, device=DEVICE)
    MODEL_SAVE_PATH = f"simclr_encoder_action_dim{ACTION_DIM}_gpu.pth"

    print(f"Starting SimCLR pre-training for {NUM_EPOCHS_PRETRAIN} epochs on {DEVICE}...")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS_PRETRAIN):
        epoch_loss_sum = 0.0
        simclr_model.train()
        for views1_tokens, views2_tokens in train_loader:
            views1_tokens, views2_tokens = views1_tokens.to(DEVICE), views2_tokens.to(DEVICE)

            optimizer.zero_grad()
            z1, z2 = simclr_model(views1_tokens), simclr_model(views2_tokens)
            z_all = torch.cat([z1, z2], dim=0)

            loss = criterion(z_all)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss_sum += loss.item()

        avg_loss = epoch_loss_sum / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS_PRETRAIN}], Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    print(f"Pre-training finished in {time.time() - start_time:.2f} seconds.")
    torch.save(simclr_model.base_encoder.state_dict(), MODEL_SAVE_PATH)
    print(f"Pre-trained base encoder saved to {MODEL_SAVE_PATH}")

    # --- Testing Phase ---
    print("\n--- Loading base encoder for testing ---")
    loaded_base_encoder = ConstraintEmbeddingTransformer(
        action_dim=ACTION_DIM, d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=D_MODEL * 2, dropout=0.1
    )
    try:
        loaded_base_encoder.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"Successfully loaded pre-trained base encoder from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}. Using random encoder for testing.")

    loaded_base_encoder.to(DEVICE).eval()

    # MODIFIED FOR HOPPER: New 3D constraints for similarity test
    print("\n--- Running Similarity Test (3D Adapted for Hopper) ---")
    raw_sphere_A1 = {
        'type': 'l2_norm', 'name': 'Sphere_A1', 'action_dim': ACTION_DIM,
        'params': {'center': np.zeros(ACTION_DIM), 'radius': 0.5}
    }
    raw_sphere_A2 = {
        'type': 'l2_norm', 'name': 'Sphere_A2_Similar', 'action_dim': ACTION_DIM,
        'params': {'center': np.array([0.1, -0.1, 0.05]), 'radius': 0.45}
    }
    raw_box_B = {
        'type': 'box', 'name': 'Box_B_Dissimilar', 'action_dim': ACTION_DIM,
        'params': {'low': np.array([1.0, 1.0, 1.0]), 'high': np.array([1.5, 1.5, 1.5])}
    }
    raw_union_A_like = {
        'type': 'union', 'name': 'Union_A1_like', 'action_dim': ACTION_DIM,
        'params': {'components': [
            {'type': 'l2_norm', 'params': {'center': np.zeros(ACTION_DIM), 'radius': 0.5}, 'action_dim': ACTION_DIM},
            {'type': 'box', 'params': {'low': np.array([0.4, 0.4, 0.4]), 'high': np.array([0.45, 0.45, 0.45])},
             'action_dim': ACTION_DIM}
        ]}
    }
    test_constraints_for_sim_test = [raw_sphere_A1, raw_sphere_A2, raw_box_B, raw_union_A_like]

    print("Generating embeddings for similarity test constraints...")
    test_embeddings = get_embeddings_for_inference(
        loaded_base_encoder, test_constraints_for_sim_test,
        N_SAMPLES_PER_VIEW, ACTION_DIM, SAMPLING_RANGE, DEVICE
    )

    if test_embeddings.shape[0] == len(test_constraints_for_sim_test):
        emb_A1, emb_A2, emb_B, emb_Union_A_like = test_embeddings[0], test_embeddings[1], test_embeddings[2], \
        test_embeddings[3]

        def calculate_cosine_sim(v1, v2):
            return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

        sim_A1_A2 = calculate_cosine_sim(emb_A1, emb_A2)
        sim_A1_B = calculate_cosine_sim(emb_A1, emb_B)
        sim_A1_Union = calculate_cosine_sim(emb_A1, emb_Union_A_like)

        print(f"\n--- Cosine Similarities (Action Dim: {ACTION_DIM}) ---")
        print(f"Sim(Sphere_A1, Sphere_A2_Similar): {sim_A1_A2:.4f} (Expected: High)")
        print(f"Sim(Sphere_A1, Box_B_Dissimilar): {sim_A1_B:.4f} (Expected: Low)")
        print(f"Sim(Sphere_A1, Union_A1_like): {sim_A1_Union:.4f} (Expected: High)")
    else:
        print("Could not generate all test embeddings.")

    print("\n--- Test Finished ---")


if __name__ == '__main__':
    main_pretrain_and_test()