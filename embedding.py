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
        # For a batch [z1_1, z1_2, ..., z1_M, z2_1, z2_2, ..., z2_M]
        # Positive for z1_i (index i) is z2_i (index i + M)
        # Positive for z2_i (index i + M) is z1_i (index i)
        self.positive_pair_indices = torch.cat([
            labels[num_original_constraints:],
            labels[:num_original_constraints]
        ])
        self.identity_mask = (~torch.eye(batch_size_effective, dtype=torch.bool, device=self.device))

    def forward(self, z_all: torch.Tensor) -> torch.Tensor:
        z_all_norm = F.normalize(z_all, dim=1)
        sim_matrix = torch.mm(z_all_norm, z_all_norm.t())
        sim_matrix_scaled = sim_matrix / self.temperature

        # Get similarities of positive pairs using the precomputed indices
        sim_positive = sim_matrix_scaled[torch.arange(z_all.shape[0], device=self.device), self.positive_pair_indices]
        numerator = torch.exp(sim_positive)

        # Denominator: Sum over all k != i of exp(sim(i, k) / T)
        # Mask out self-correlations (diagonal) for negative examples sum
        exp_sim_negatives = torch.exp(sim_matrix_scaled) * self.identity_mask
        denominator = torch.sum(exp_sim_negatives, dim=1)

        loss = -torch.log(numerator / (denominator + 1e-8))  # Add epsilon for stability
        return loss.mean()


# --- 1. ConstraintEmbeddingTransformer (Base Encoder f(·)) ---
class ConstraintEmbeddingTransformer(nn.Module):
    def __init__(self, action_dim: int, d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model

        self.input_token_dim = action_dim + 1
        self.linear_projection = nn.Linear(self.input_token_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation=F.gelu
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, constraint_tokens_batch: torch.Tensor) -> torch.Tensor:
        batch_size = constraint_tokens_batch.shape[0]
        projected_tokens = self.linear_projection(constraint_tokens_batch)
        S = torch.cat([self.cls_token.expand(batch_size, -1, -1), projected_tokens], dim=1)
        transformer_output = self.transformer_encoder(S)
        cls_output_h_prime = transformer_output[:, 0, :]  # Output from CLS token, [batch_size, d_model]
        return cls_output_h_prime


# --- 2. SimCLR Model Wrapper (Base Encoder + Projection Head g(·)) ---
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder: ConstraintEmbeddingTransformer, projection_dim: int = 128):
        super().__init__()
        self.base_encoder = base_encoder
        h_prime_dim = base_encoder.d_model  # Dimension of the output from base_encoder

        self.projection_head = nn.Sequential(
            nn.Linear(h_prime_dim, h_prime_dim),  # Hidden layer in projection head
            nn.ReLU(),
            nn.Linear(h_prime_dim, projection_dim)  # Output 'z' for contrastive loss
        )

    def forward(self, constraint_tokens_batch: torch.Tensor) -> torch.Tensor:
        h_prime = self.base_encoder(constraint_tokens_batch)
        z = self.projection_head(h_prime)
        return z


# --- 3. `cfunc` Implementations & Dispatcher ---
ConstraintFunction = Callable[[torch.Tensor], torch.Tensor]


def cfunc_l2_norm(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    return torch.linalg.norm(x - center, ord=2, dim=-1) - radius


def cfunc_l1_norm(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    return torch.linalg.norm(x - center, ord=1, dim=-1) - radius


def cfunc_linear_halfspace(x: torch.Tensor, a: torch.Tensor, b: float) -> torch.Tensor:
    # Ensure a is a 1D vector for dot product if x is 1D, or correctly shaped for matmul if x is 2D
    if x.ndim == 1 and a.ndim > 1 and a.shape[0] == x.shape[0]:  # a might be (D,1) or (1,D) but x is (D)
        a = a.squeeze()
    elif x.ndim > 1 and a.ndim == 1:  # x is (B,D) a is (D)
        a = a.unsqueeze(0)  # make a (1,D) for broadcasting with matmul or make it (D,1) for x@a.T
        return torch.matmul(x, a.t()).squeeze(-1) - b  # x (B,D) @ a (D,1) -> (B,1) -> (B)
    return torch.matmul(x, a) - b  # Assumes x (B,D) and a (D), or x (D) and a (D)


def cfunc_box(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    # Violations for each boundary: x_i - high_i (should be <=0) and low_i - x_i (should be <=0)
    # We want a single c_i value. max_violation <= 0 means feasible.
    if x.ndim == 1:  # Single point (action_dim)
        violations = torch.cat([(x - high), (low - x)])  # (2*action_dim)
    else:  # Batch of points (batch_size, action_dim)
        violations = torch.cat([(x - high), (low - x)], dim=1)  # (batch_size, 2*action_dim)
    return torch.max(violations, dim=-1).values


def cfunc_polytope(x: torch.Tensor, A_matrix: torch.Tensor, b_vector: torch.Tensor) -> torch.Tensor:
    # Ax <= b  => Ax - b <= 0.  Want max_i((Ax)_i - b_i)
    if x.ndim == 1:  # x: (D), A: (M, D), b: (M)
        violations = torch.mv(A_matrix, x) - b_vector  # (M)
    else:  # x: (B, D), A: (M, D), b: (M)
        violations = torch.matmul(x, A_matrix.t()) - b_vector.unsqueeze(0)  # (B, M)
    return torch.max(violations, dim=-1).values


def cfunc_ellipsoid(x: torch.Tensor, center: torch.Tensor, P_inv: torch.Tensor) -> torch.Tensor:
    # (x-c)^T P_inv (x-c) <= 1
    diff = x - center
    if diff.ndim == 1:  # diff: (D), P_inv: (D, D)
        # (D) @ (D) element-wise with (D)
        return torch.mv(P_inv, diff).dot(diff) - 1.0
    else:  # diff: (B, D), P_inv: (D, D)
        # (B,D) @ (D,D) -> (B,D). Then element-wise with (B,D) and sum over D
        # P_inv @ diff.unsqueeze(-1) -> (D,1) for each row, then diff @ result
        # Simpler: torch.einsum('bi,ij,bj->b', diff, P_inv, diff)
        return torch.sum(torch.matmul(diff, P_inv) * diff, dim=-1) - 1.0


def cfunc_soc(x: torch.Tensor, A_soc: torch.Tensor, b_soc: torch.Tensor, c_soc: torch.Tensor,
              d_soc: float) -> torch.Tensor:
    # ||Ax+b||_2 <= c^T x + d
    if x.ndim == 1:  # x: (D), A_soc: (M,D), b_soc: (M), c_soc: (D)
        Ax_plus_b = torch.mv(A_soc, x) + b_soc  # (M)
        cTx_plus_d = torch.dot(c_soc, x) + d_soc  # scalar
    else:  # x: (B,D)
        Ax_plus_b = torch.matmul(x, A_soc.t()) + b_soc.unsqueeze(0)  # (B,M)
        cTx_plus_d = torch.matmul(x, c_soc.unsqueeze(-1)).squeeze(-1) + d_soc  # (B)

    norm_Ax_plus_b = torch.linalg.norm(Ax_plus_b, ord=2, dim=-1)  # (B) or scalar
    return norm_Ax_plus_b - cTx_plus_d


class CallableOracleWrapper:
    def __init__(self, oracle: Callable[[torch.Tensor], Any], device: torch.device):
        self.oracle = oracle
        self.device = device

    def __call__(self, x_i: torch.Tensor) -> torch.Tensor:
        val = self.oracle(x_i.to(self.device))  # x_i already on device from _generate_one_view
        if not isinstance(val, torch.Tensor):
            val = torch.tensor([val], dtype=torch.float32, device=self.device)
        elif val.ndim == 0:
            val = val.unsqueeze(0)
        return val.float()


def build_cfunc_from_raw_data(raw_data: Dict[str, Any], device: torch.device) -> ConstraintFunction:
    ctype = raw_data['type']
    params = raw_data.get('params', {})

    processed_params = {}
    # Critical: ensure P_inv for ellipsoid is (D,D) and A for polytope is (M,D) etc.
    # The generate_training_raw_data should produce them in correct numpy shapes.
    for key, val in params.items():
        if key == 'components' and ctype == 'union':
            processed_params[key] = val  # List of dicts, will be processed recursively
        elif isinstance(val, (np.ndarray, list)):
            # Ensure correct dtype before creating tensor
            if isinstance(val, list) and all(isinstance(i, (int, float)) for i in val):  # simple list of numbers
                val_np = np.array(val, dtype=np.float32)
            elif isinstance(val, list) and all(isinstance(i, list) for i in val):  # list of lists (matrix)
                val_np = np.array(val, dtype=np.float32)
            elif isinstance(val, np.ndarray):
                val_np = val.astype(np.float32) if val.dtype != np.float32 else val
            else:  # Mixed list, or other complex structure not directly convertible to simple tensor
                # This branch might need more specific handling based on your data
                print(
                    f"Warning: Parameter '{key}' for type '{ctype}' has complex list structure: {val}. Attempting direct tensor conversion.")
                val_np = np.array(val, dtype=np.float32)  # May fail or misinterpret

            processed_params[key] = torch.from_numpy(val_np).to(device)

        elif isinstance(val, torch.Tensor):
            processed_params[key] = val.to(dtype=torch.float32, device=device)
        else:  # floats, ints
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
        component_cfuncs = [build_cfunc_from_raw_data(comp_raw_data, device) for comp_raw_data in
                            processed_params['components']]

        def cfunc_union(x: torch.Tensor) -> torch.Tensor:
            # Handle batch of x for cfunc_union
            all_component_evals = []
            for cf in component_cfuncs:
                eval_ = cf(x)  # cf should handle batched x and return (batch_size,) or (1,)
                if x.ndim > 1 and eval_.ndim == 0:  # Batched x but cfunc returned scalar
                    eval_ = eval_.expand(x.shape[0])
                all_component_evals.append(eval_)

            component_evals_stacked = torch.stack(all_component_evals,
                                                  dim=0)  # (num_components, batch_size) or (num_components)

            if component_evals_stacked.ndim > 1:  # if result is (num_components, batch_size)
                return torch.min(component_evals_stacked, dim=0).values  # min over components for each x
            else:  # result is (num_components)
                return torch.min(component_evals_stacked)

        return cfunc_union
    elif callable(raw_data.get('cfunc_oracle')):
        return CallableOracleWrapper(raw_data['cfunc_oracle'], device)
    else:
        raise ValueError(f"Unsupported constraint type or missing cfunc_oracle: {ctype}")


# --- 4. Constraint Dataset for SimCLR ---
class ConstraintDatasetSimCLR(Dataset):
    def __init__(self, list_of_constraint_raw_data: List[Dict[str, Any]],
                 N_samples_per_view: int, action_dim: int,
                 sampling_range: Tuple[float, float] = (-2.0, 2.0),
                 device_for_cfunc: torch.device = torch.device("cpu")):  # Device for cfunc computations
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
            # x_i is already on self.device_for_cfunc

            c_i_val = cfunc(x_i)
            if not isinstance(c_i_val, torch.Tensor):
                c_i_val = torch.tensor([c_i_val], dtype=torch.float32, device=self.device_for_cfunc)
            # Ensure c_i is scalar-like (1 element) before cat
            if c_i_val.numel() != 1:
                # This case should ideally not happen if cfuncs are defined correctly to return scalar constraint value
                print(f"Warning: c_i_val for {raw_data.get('name')} not scalar: {c_i_val}. Taking first element.")
                c_i_val = c_i_val.flatten()[0]
            c_i = c_i_val.reshape(1).float()  # Ensure (1,) shape and float

            tokens.append(torch.cat([x_i, c_i]))
        return torch.stack(tokens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_data = self.constraint_raw_data_list[idx]
        view1_tokens = self._generate_one_view(raw_data)
        view2_tokens = self._generate_one_view(raw_data)  # Independent sampling
        return view1_tokens, view2_tokens


# --- 5. Helper: Generate Training Data (Dummy) ---
def generate_training_raw_data(num_total: int, action_dim: int) -> List[Dict[str, Any]]:
    all_raw_data = []
    for i in range(num_total):
        types = ['l2_norm', 'box', 'linear_halfspace']
        if action_dim >= 2:
            types.extend(['polytope', 'ellipsoid', 'soc', 'union'])
        chosen_type = np.random.choice(types)
        constraint_raw = {'name': f"{chosen_type}_{i}", 'type': chosen_type}

        params = {}
        if chosen_type == 'l2_norm':
            params = {'center': np.random.uniform(-1, 1, size=action_dim), 'radius': np.random.uniform(0.5, 1.5)}
        elif chosen_type == 'box':
            low = np.random.uniform(-1.5, 0, size=action_dim);
            high = low + np.random.uniform(0.5, 1.5, size=action_dim)
            params = {'low': low, 'high': high}
        elif chosen_type == 'linear_halfspace':
            a_vec = np.random.randn(action_dim)
            # Normalize 'a' to prevent very large/small values if desired
            # a_vec = a_vec / np.linalg.norm(a_vec) if np.linalg.norm(a_vec) > 1e-6 else a_vec
            params = {'a': a_vec, 'b': np.random.randn() * 0.5}
        elif chosen_type == 'polytope':
            num_halfspaces = np.random.randint(action_dim + 1, action_dim + 3)
            A = np.random.randn(num_halfspaces, action_dim)
            # Define b such that origin is likely feasible for non-trivial polytope
            b = np.random.rand(num_halfspaces) * 0.5 + 0.1  # Ax <= b, so b should be positive-ish
            params = {'A': A, 'b': b}
        elif chosen_type == 'ellipsoid':
            center = np.random.uniform(-1, 1, size=action_dim)
            # Random positive definite P_inv: L L^T, where L is lower triangular with positive diagonal
            L = np.random.rand(action_dim, action_dim) * 0.5
            for d_idx in range(action_dim): L[d_idx, d_idx] = np.random.uniform(0.5, 1.5)  # Ensure positive diag
            L[np.triu_indices(action_dim, k=1)] = 0  # Make L lower triangular
            P_inv = L @ L.T + np.eye(action_dim) * 1e-3  # Add small identity for numerical stability
            params = {'center': center, 'P_inv': P_inv}
        elif chosen_type == 'soc':
            m_soc = max(1, action_dim)  # Dimension of A_soc output
            A_s = np.random.randn(m_soc, action_dim) * 0.5
            b_s = np.random.randn(m_soc) * 0.5
            c_s = np.random.randn(action_dim) * 0.1
            d_s = np.random.uniform(np.linalg.norm(b_s) + 0.1,
                                    np.linalg.norm(b_s) + 1.0)  # Ensure feasibility around origin
            params = {'A_soc': A_s, 'b_soc': b_s, 'c_soc': c_s, 'd_soc': d_s}
        elif chosen_type == 'union':
            num_components = 2;
            components = []
            for _ in range(num_components):
                comp_type = np.random.choice(['l2_norm', 'box'])
                comp_raw_child = {'name': f"{comp_type}_comp", 'type': comp_type}
                child_params = {}
                if comp_type == 'l2_norm':
                    child_params = {'center': np.random.uniform(-1.5, 1.5, size=action_dim),
                                    'radius': np.random.uniform(0.2, 0.5)}
                else:  # box
                    low = np.random.uniform(-1.5, 1, size=action_dim);
                    high = low + np.random.uniform(0.2, 0.5, size=action_dim)
                    child_params = {'low': low, 'high': high}
                comp_raw_child['params'] = child_params
                components.append(comp_raw_child)
            params = {'components': components}
        constraint_raw['params'] = params
        all_raw_data.append(constraint_raw)
    return all_raw_data


# --- 6. Helper: Get Embeddings for Inference/Testing ---
def get_embeddings_for_inference(
        base_model: ConstraintEmbeddingTransformer,
        list_of_raw_data: List[Dict[str, Any]],
        N_samples: int, action_dim: int, sampling_range: Tuple[float, float], device: torch.device
) -> torch.Tensor:
    base_model.eval()
    base_model.to(device)
    batch_tokens_list = []

    # Use a temporary dataset to leverage its _generate_one_view, which handles cfunc creation
    temp_dataset = ConstraintDatasetSimCLR(
        list_of_constraint_raw_data=list_of_raw_data,  # Pass the list to be processed
        N_samples_per_view=N_samples,
        action_dim=action_dim,
        sampling_range=sampling_range,
        device_for_cfunc=device
    )

    for i in range(len(list_of_raw_data)):  # Iterate based on the input list
        # _generate_one_view needs the raw_data item, not just index
        raw_data_item = list_of_raw_data[i]
        view_tokens = temp_dataset._generate_one_view(raw_data_item)  # Generate one view
        batch_tokens_list.append(view_tokens)

    if not batch_tokens_list:
        return torch.empty(0, base_model.d_model, device=device)

    batch_tokens_tensor = torch.stack(batch_tokens_list).to(device)
    with torch.no_grad():
        h_prime_embeddings = base_model(batch_tokens_tensor)  # Output of base encoder
    return h_prime_embeddings


# --- 7. Main Pre-training and Testing Script ---
def main_cpu_small_pretrain_and_test():
    print("--- Script Starting: main_cpu_small_pretrain_and_test ---")
    # --- Parameters for Small CPU Test ---
    ACTION_DIM = 2
    D_MODEL = 32
    N_HEAD = 2
    NUM_ENCODER_LAYERS = 2
    PROJECTION_DIM = 64
    TEMPERATURE = 0.2

    N_SAMPLES_PER_VIEW = 30
    SAMPLING_RANGE = (-2.0, 2.0)

    NUM_EPOCHS_PRETRAIN = 1000  # Reduced for quick CPU test
    BATCH_SIZE_CONSTRAINTS = 4
    if BATCH_SIZE_CONSTRAINTS < 2 and NUM_EPOCHS_PRETRAIN > 0:
        print("Warning: BATCH_SIZE_CONSTRAINTS for SimCLR pretraining should be >= 2. Adjusting to 2.")
        BATCH_SIZE_CONSTRAINTS = 2

    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5

    DEVICE = torch.device("cpu")
    print(f"Running on device: {DEVICE}")

    # --- Generate Small Dummy Constraint Dataset ---
    num_training_constraints = BATCH_SIZE_CONSTRAINTS * 5
    if num_training_constraints < BATCH_SIZE_CONSTRAINTS and NUM_EPOCHS_PRETRAIN > 0:
        num_training_constraints = BATCH_SIZE_CONSTRAINTS
    print(f"Generating {num_training_constraints} dummy constraints for pre-training...")
    training_constraint_definitions = generate_training_raw_data(num_training_constraints, ACTION_DIM)
    print(f"Number of training definitions generated: {len(training_constraint_definitions)}")

    if not training_constraint_definitions and NUM_EPOCHS_PRETRAIN > 0:
        print("No training data generated. Skipping pre-training and testing.")
        return

    train_dataset = ConstraintDatasetSimCLR(
        list_of_constraint_raw_data=training_constraint_definitions,
        N_samples_per_view=N_SAMPLES_PER_VIEW, action_dim=ACTION_DIM,
        sampling_range=SAMPLING_RANGE, device_for_cfunc=DEVICE
    )
    # drop_last=True is important if NT_Xent_Loss expects a fixed batch size
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_CONSTRAINTS, shuffle=True,
        num_workers=0, drop_last=True if len(training_constraint_definitions) >= BATCH_SIZE_CONSTRAINTS else False,
        pin_memory=False
    )
    print(f"Length of train_loader: {len(train_loader)}")

    # --- Model, Optimizer, Loss ---
    base_encoder = ConstraintEmbeddingTransformer(
        action_dim=ACTION_DIM, d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=D_MODEL * 2, dropout=0.1
    )
    simclr_model = SimCLRModel(base_encoder, projection_dim=PROJECTION_DIM).to(DEVICE)
    optimizer = optim.AdamW(simclr_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # CosineAnnealingLR needs T_max > 0 if NUM_EPOCHS_PRETRAIN > 0
    scheduler = None
    if NUM_EPOCHS_PRETRAIN > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_PRETRAIN, eta_min=0)

    criterion = NT_Xent_Loss(batch_size_effective=BATCH_SIZE_CONSTRAINTS * 2, temperature=TEMPERATURE, device=DEVICE)

    MODEL_SAVE_PATH = "cpu_small_simclr_encoder_base.pth"
    # --- Training Loop ---
    if NUM_EPOCHS_PRETRAIN > 0 and len(train_loader) > 0:
        print(f"Starting SimCLR pre-training for {NUM_EPOCHS_PRETRAIN} epochs on CPU (this might be slow)...")
        start_time = time.time()
        for epoch in range(NUM_EPOCHS_PRETRAIN):
            epoch_loss_sum = 0.0
            batches_in_epoch = 0
            simclr_model.train()
            for i, (views1_tokens, views2_tokens) in enumerate(train_loader):
                if views1_tokens.shape[0] < BATCH_SIZE_CONSTRAINTS and train_loader.drop_last == False:
                    print(
                        f"Skipping last batch of size {views1_tokens.shape[0]} as it's smaller than BATCH_SIZE_CONSTRAINTS and drop_last is False.")
                    continue  # NTXent needs full batch if not handled specifically

                views1_tokens = views1_tokens.to(DEVICE)
                views2_tokens = views2_tokens.to(DEVICE)
                optimizer.zero_grad()
                z1 = simclr_model(views1_tokens)
                z2 = simclr_model(views2_tokens)
                z_all = torch.cat([z1, z2], dim=0)

                # Ensure z_all has the expected batch size for NT_Xent_Loss
                if z_all.shape[0] != criterion.batch_size_effective:
                    # This can happen if the last batch is smaller and drop_last=False
                    # For simplicity, we skip such batches if drop_last wasn't effective or set.
                    # A more robust NT_Xent_Loss could handle variable batch sizes.
                    print(
                        f"Warning: Effective batch size {z_all.shape[0]} for NTXent does not match expected {criterion.batch_size_effective}. Skipping batch.")
                    continue

                loss = criterion(z_all)
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.item()
                batches_in_epoch += 1

            if scheduler:
                scheduler.step()

            if batches_in_epoch > 0:
                avg_epoch_loss = epoch_loss_sum / batches_in_epoch
                current_lr_msg = f", LR: {scheduler.get_last_lr()[0]:.6f}" if scheduler else ""
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS_PRETRAIN}], Avg Loss: {avg_epoch_loss:.4f}{current_lr_msg}")
            else:
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS_PRETRAIN}]: No batches were processed.")

        end_time = time.time()
        print(f"Pre-training finished in {end_time - start_time:.2f} seconds.")

        torch.save(simclr_model.base_encoder.state_dict(), MODEL_SAVE_PATH)
        print(f"Pre-trained base encoder saved to {MODEL_SAVE_PATH}")
    elif len(train_loader) == 0 and NUM_EPOCHS_PRETRAIN > 0:
        print("No training data loaded (train_loader is empty). Skipping pre-training.")
    else:  # NUM_EPOCHS_PRETRAIN == 0
        print("NUM_EPOCHS_PRETRAIN is 0. Skipping pre-training.")

    # --- Load the pre-trained (or randomly initialized if no training) base encoder ---
    print("\n--- Loading base encoder for testing ---")
    loaded_base_encoder = ConstraintEmbeddingTransformer(
        action_dim=ACTION_DIM, d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=D_MODEL * 2, dropout=0.1
    )
    model_was_loaded = False
    if NUM_EPOCHS_PRETRAIN > 0 and (len(train_loader) > 0 if train_loader else False):  # Check if training actually ran
        try:
            loaded_base_encoder.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
            print(f"Successfully loaded pre-trained base encoder from {MODEL_SAVE_PATH}")
            model_was_loaded = True
        except FileNotFoundError:
            print(f"Model file {MODEL_SAVE_PATH} not found. Using randomly initialized base encoder for testing.")
        except Exception as e:
            print(f"Error loading model: {e}. Using randomly initialized base encoder.")
    else:
        print("Using randomly initialized base encoder for testing (no pre-training done or model not saved).")
    loaded_base_encoder.to(DEVICE).eval()

    # --- Similarity Test ---
    print("\n--- Running Similarity Test ---")
    raw_sphere_A1 = {'type': 'l2_norm', 'name': 'SphereA1', 'params': {'center': np.array([0.0, 0.0]), 'radius': 1.0}}
    raw_sphere_A2 = {'type': 'l2_norm', 'name': 'SphereA2_Similar',
                     'params': {'center': np.array([0.1, -0.1]), 'radius': 0.9}}
    raw_box_B = {'type': 'box', 'name': 'BoxB_Dissimilar',
                 'params': {'low': np.array([-2.0, -2.0]), 'high': np.array([-1.0, -1.0])}}
    raw_union_A_like = {
        'type': 'union', 'name': 'Union_SphereA1_like',
        'params': {'components': [
            {'type': 'l2_norm', 'params': {'center': np.array([0.0, 0.0]), 'radius': 1.0}},
            {'type': 'box', 'params': {'low': np.array([0.7, 0.7]), 'high': np.array([0.8, 0.8])}}
        ]}
    }
    test_constraints_for_sim_test = [raw_sphere_A1, raw_sphere_A2, raw_box_B, raw_union_A_like]

    print("Generating embeddings for similarity test constraints...")
    test_embeddings = get_embeddings_for_inference(
        loaded_base_encoder, test_constraints_for_sim_test,
        N_SAMPLES_PER_VIEW, ACTION_DIM, SAMPLING_RANGE, DEVICE
    )

    if test_embeddings.shape[0] == len(test_constraints_for_sim_test):
        emb_A1 = test_embeddings[0];
        emb_A2 = test_embeddings[1]
        emb_B = test_embeddings[2];
        emb_Union_A_like = test_embeddings[3]

        def calculate_cosine_sim(v1, v2):
            return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

        sim_A1_A2 = calculate_cosine_sim(emb_A1, emb_A2)
        sim_A1_B = calculate_cosine_sim(emb_A1, emb_B)
        sim_A2_B = calculate_cosine_sim(emb_A2, emb_B)
        sim_A1_Union = calculate_cosine_sim(emb_A1, emb_Union_A_like)

        print(f"\n--- Cosine Similarities ---")
        print(f"Similarity(SphereA1, SphereA2_Similar): {sim_A1_A2:.4f} (Expected: High if model learned)")
        print(f"Similarity(SphereA1, BoxB_Dissimilar): {sim_A1_B:.4f} (Expected: Low if model learned)")
        print(f"Similarity(SphereA2_Similar, BoxB_Dissimilar): {sim_A2_B:.4f} (Expected: Low if model learned)")
        print(
            f"Similarity(SphereA1, Union_SphereA1_like): {sim_A1_Union:.4f} (Expected: Reasonably High if model learned)")
    else:
        print(
            f"Could not generate embeddings for all test constraints. Embeddings shape: {test_embeddings.shape}, Expected: {len(test_constraints_for_sim_test)}")

    print("\n--- Test Finished ---")
    if model_was_loaded:
        print(
            "Note: With very few epochs and small batch size on CPU, learned similarities might not be very strong or indicative of true performance.")
    else:
        print(
            "Note: Embeddings were generated by a RANDOMLY INITIALIZED model as no pre-training was loaded or completed successfully.")
    print("This script is primarily for testing the workflow and structure on CPU.")


if __name__ == '__main__':
    print("--- Script execution started (embedding.py) ---")
    main_cpu_small_pretrain_and_test()
    print("--- Script execution normally finished (embedding.py) ---")