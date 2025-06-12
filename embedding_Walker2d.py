import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Callable, Dict, Any, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import time
import os

# 确保已安装Gymnasium: pip install gymnasium[mujoco]
try:
    import gymnasium as gym
except ImportError:
    print("警告: Gymnasium未安装。将使用硬编码的Walker2d维度。请运行 `pip install gymnasium[mujoco]`")
    # 如果gymnasium未安装，手动定义维度
    WALKER_STATE_DIM = 17
    WALKER_ACTION_DIM = 6


# --- 0. NT-Xent Loss (无需改动) ---
# 这个损失函数是通用的，不需要修改。
class NT_Xent_Loss(nn.Module):
    def __init__(self, batch_size_effective: int, temperature: float, device: torch.device):
        super(NT_Xent_Loss, self).__init__()
        self.batch_size_effective = batch_size_effective
        self.temperature = temperature
        self.device = device

        num_original_constraints = batch_size_effective // 2
        labels = torch.arange(batch_size_effective, device=self.device)
        self.positive_pair_indices = torch.cat([
            labels[num_original_constraints:],
            labels[:num_original_constraints]
        ])
        self.identity_mask = (~torch.eye(batch_size_effective, dtype=torch.bool, device=self.device))

    def forward(self, z_all: torch.Tensor) -> torch.Tensor:
        z_all_norm = F.normalize(z_all, dim=1)
        sim_matrix = torch.mm(z_all_norm, z_all_norm.t())
        sim_matrix_scaled = sim_matrix / self.temperature
        sim_positive = sim_matrix_scaled[torch.arange(z_all.shape[0], device=self.device), self.positive_pair_indices]
        numerator = torch.exp(sim_positive)
        exp_sim_negatives = torch.exp(sim_matrix_scaled) * self.identity_mask
        denominator = torch.sum(exp_sim_negatives, dim=1)
        loss = -torch.log(numerator / (denominator + 1e-8))
        return loss.mean()


# --- 1. TaskEmbeddingTransformer (无需改动) ---
# 此模块是通用的，接收状态和动作维度作为参数。
class TaskEmbeddingTransformer(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError(f"state_dim和action_dim必须为正，得到 {state_dim}, {action_dim}")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Token现在是状态和动作向量的拼接
        self.input_token_dim = state_dim + action_dim

        self.linear_projection = nn.Linear(self.input_token_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation=F.gelu
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, task_tokens_batch: torch.Tensor) -> torch.Tensor:
        if task_tokens_batch.shape[-1] != self.input_token_dim:
            raise ValueError(f"输入token维度不匹配。期望 {self.input_token_dim}, "
                             f"但得到 {task_tokens_batch.shape[-1]}")
        batch_size = task_tokens_batch.shape[0]
        projected_tokens = self.linear_projection(task_tokens_batch)
        S = torch.cat([self.cls_token.expand(batch_size, -1, -1), projected_tokens], dim=1)
        transformer_output = self.transformer_encoder(S)
        cls_output_h_prime = transformer_output[:, 0, :]
        return cls_output_h_prime


# --- 2. SimCLR Model Wrapper (无需改动) ---
# 仅更新了类型提示，逻辑不变。
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder: TaskEmbeddingTransformer, projection_dim: int = 128):
        super().__init__()
        self.base_encoder = base_encoder
        h_prime_dim = base_encoder.d_model
        self.projection_head = nn.Sequential(
            nn.Linear(h_prime_dim, h_prime_dim),
            nn.ReLU(),
            nn.Linear(h_prime_dim, projection_dim)
        )

    def forward(self, task_tokens_batch: torch.Tensor) -> torch.Tensor:
        h_prime = self.base_encoder(task_tokens_batch)
        z = self.projection_head(h_prime)
        return z


# --- 3. MuJoCo任务数据集 (NEW, 通用化) ---
# 从HopperTaskDataset重命名，使其更通用。
class MujocoTaskDataset(Dataset):
    def __init__(self,
                 task_definitions: List[Dict[str, Any]],
                 task_data: Dict[str, np.ndarray],
                 samples_per_view: int):
        self.task_definitions = task_definitions
        self.task_data = task_data
        self.samples_per_view = samples_per_view
        self.task_names = [t['name'] for t in self.task_definitions]

    def __len__(self):
        return len(self.task_definitions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        task_name = self.task_names[idx]
        trajectory = self.task_data[task_name]

        num_timesteps = trajectory.shape[0]
        if num_timesteps < self.samples_per_view:
            raise ValueError(f"任务 {task_name} 只有 {num_timesteps} 个时间步, "
                             f"但每个视图需要 {self.samples_per_view} 个。")

        # 通过从同一轨迹中采样不同的索引来创建两个视图
        indices1 = np.random.choice(num_timesteps, self.samples_per_view, replace=False)
        indices2 = np.random.choice(num_timesteps, self.samples_per_view, replace=False)

        view1_tokens = torch.from_numpy(trajectory[indices1]).float()
        view2_tokens = torch.from_numpy(trajectory[indices2]).float()

        return view1_tokens, view2_tokens


# --- 4. 辅助函数: 生成MuJoCo训练数据 (NEW, 通用化) ---
# 此函数定义MuJoCo任务并创建*虚拟*轨迹数据。
def generate_mujoco_task_definitions_and_data(
        num_tasks: int,
        timesteps_per_task: int,
        state_dim: int,
        action_dim: int,
        env_name: str = "Walker2d"
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    生成任务定义列表和相应的虚拟轨迹数据。
    在真实应用中，您应该加载由专家策略（例如用SAC或PPO训练）生成的真实轨迹。
    """
    print(f"正在为 {env_name} 生成 {num_tasks} 个虚拟任务，每个任务 {timesteps_per_task} 个时间步...")

    task_definitions = []
    task_data = {}

    # 任务：以不同的前向速度奔跑
    target_velocities = np.linspace(0.5, 3.0, num_tasks)  # Walker2d可以跑得更快

    for i, vel in enumerate(target_velocities):
        task_name = f"{env_name}_target_vel_{vel:.2f}"
        task_def = {
            'name': task_name,
            'type': 'target_velocity',
            'params': {'velocity': vel}
        }
        task_definitions.append(task_def)

        # --- 虚拟数据生成 ---
        # 在真实项目中，用加载真实轨迹替换此处。
        # 例如: `trajectory = np.load(f'trajectories/{task_name}.npy')`
        # 数据应该是来自专家策略的(状态, 动作)对。
        dummy_states = np.random.randn(timesteps_per_task, state_dim)
        dummy_actions = np.random.uniform(-1.0, 1.0, size=(timesteps_per_task, action_dim))

        # 通过平移使每个任务的数据略有不同
        dummy_states[:, 0] += vel  # 假设x轴速度与目标速度相关

        trajectory = np.concatenate([dummy_states, dummy_actions], axis=1).astype(np.float32)
        task_data[task_name] = trajectory
        # --- 结束虚拟数据 ---

    print(f"生成了 {len(task_definitions)} 个任务定义和数据。")
    return task_definitions, task_data


# --- 5. 辅助函数: 获取用于推理的嵌入 (无需改动) ---
def get_embeddings_for_inference(
        base_model: TaskEmbeddingTransformer,
        task_definitions: List[Dict[str, Any]],
        task_data: Dict[str, np.ndarray],
        samples_per_view: int,
        device: torch.device
) -> torch.Tensor:
    base_model.eval()
    base_model.to(device)

    inference_dataset = MujocoTaskDataset(
        task_definitions, task_data, samples_per_view
    )

    batch_tokens_list = []
    for i in range(len(inference_dataset)):
        # 推理只需要一个视图
        view1_tokens, _ = inference_dataset[i]
        batch_tokens_list.append(view1_tokens)

    if not batch_tokens_list:
        return torch.empty(0, base_model.d_model, device=device)

    batch_tokens_tensor = torch.stack(batch_tokens_list).to(device)
    with torch.no_grad():
        h_prime_embeddings = base_model(batch_tokens_tensor)

    return h_prime_embeddings


# --- 6. 针对Walker2d的主预训练和测试脚本 (MODIFIED) ---
def main_pretrain_and_test_walker():
    print("--- 脚本启动: Walker2d任务嵌入预训练 ---")

    # --- Walker2d环境的参数 ---
    ENV_ID = "Walker2d-v4"
    try:
        env = gym.make(ENV_ID)
        STATE_DIM = env.observation_space.shape[0]
        ACTION_DIM = env.action_space.shape[0]
        env.close()
        print(f"找到Gymnasium。{ENV_ID} state_dim={STATE_DIM}, action_dim={ACTION_DIM}")
    except NameError:  # gym未导入
        STATE_DIM = WALKER_STATE_DIM
        ACTION_DIM = WALKER_ACTION_DIM
        print(f"未找到Gymnasium。使用硬编码的Walker2d维度: state_dim={STATE_DIM}, action_dim={ACTION_DIM}")

    D_MODEL = 256  # 对于更复杂的Walker2d，可以使用更大的模型
    N_HEAD = 8
    NUM_ENCODER_LAYERS = 6
    PROJECTION_DIM = 128
    TEMPERATURE = 0.1

    N_SAMPLES_PER_VIEW = 64
    TIMESTEPS_PER_TASK = 1000

    NUM_EPOCHS_PRETRAIN = 500  # 增加训练轮数
    BATCH_SIZE_TASKS = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"CUDA可用。使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("CUDA不可用。使用CPU。")

    NUM_WORKERS_DATALOADER = 4 if DEVICE.type == 'cuda' else 0
    PIN_MEMORY_DATALOADER = True if DEVICE.type == 'cuda' else False

    # --- 数据生成 ---
    num_training_tasks = BATCH_SIZE_TASKS * 15  # e.g., 32 * 15 = 480 个不同的任务
    training_task_defs, training_task_data = generate_mujoco_task_definitions_and_data(
        num_tasks=num_training_tasks,
        timesteps_per_task=TIMESTEPS_PER_TASK,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        env_name="Walker2d"
    )

    if not training_task_defs:
        print("未生成训练数据。退出。")
        return

    train_dataset = MujocoTaskDataset(
        task_definitions=training_task_defs,
        task_data=training_task_data,
        samples_per_view=N_SAMPLES_PER_VIEW
    )

    effective_batch_size = BATCH_SIZE_TASKS
    if len(train_dataset) < BATCH_SIZE_TASKS:
        print(f"警告: 任务数量 ({len(train_dataset)}) 小于批次大小 ({BATCH_SIZE_TASKS})。")
        effective_batch_size = len(train_dataset) if len(train_dataset) > 1 else 0

    if effective_batch_size < 2:
        print("数据不足以形成一个有效的批次。跳过训练。")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=True,
        num_workers=NUM_WORKERS_DATALOADER,
        drop_last=True,
        pin_memory=PIN_MEMORY_DATALOADER
    )

    # --- 模型, 优化器, 损失函数 ---
    base_encoder = TaskEmbeddingTransformer(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=D_MODEL * 4, dropout=0.1
    )
    simclr_model = SimCLRModel(base_encoder, projection_dim=PROJECTION_DIM).to(DEVICE)
    optimizer = optim.AdamW(simclr_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * NUM_EPOCHS_PRETRAIN)
    criterion = NT_Xent_Loss(batch_size_effective=effective_batch_size * 2, temperature=TEMPERATURE, device=DEVICE)
    MODEL_SAVE_PATH = f"simclr_walker_encoder_gpu.pth"

    # --- 训练循环 ---
    if NUM_EPOCHS_PRETRAIN > 0 and len(train_loader) > 0:
        print(f"开始在 {DEVICE} 上进行SimCLR预训练，共 {NUM_EPOCHS_PRETRAIN} 轮...")
        start_time = time.time()
        for epoch in range(NUM_EPOCHS_PRETRAIN):
            simclr_model.train()
            total_loss = 0
            for views1_tokens, views2_tokens in train_loader:
                views1_tokens = views1_tokens.to(DEVICE, non_blocking=True)
                views2_tokens = views2_tokens.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                z1 = simclr_model(views1_tokens)
                z2 = simclr_model(views2_tokens)
                z_all = torch.cat([z1, z2], dim=0)

                loss = criterion(z_all)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(
                f"轮次 [{epoch + 1}/{NUM_EPOCHS_PRETRAIN}], 平均损失: {avg_loss:.4f}, 学习率: {scheduler.get_last_lr()[0]:.6f}")

        print(f"预训练在 {time.time() - start_time:.2f} 秒内完成。")
        torch.save(simclr_model.base_encoder.state_dict(), MODEL_SAVE_PATH)
        print(f"预训练的基础编码器已保存至 {MODEL_SAVE_PATH}")
    else:
        print("跳过训练循环。")

    # --- 测试阶段 ---
    print("\n--- 加载基础编码器用于相似性测试 ---")
    loaded_base_encoder = TaskEmbeddingTransformer(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=D_MODEL * 4
    )
    model_was_loaded = False
    if os.path.exists(MODEL_SAVE_PATH):
        loaded_base_encoder.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"成功从 {MODEL_SAVE_PATH} 加载预训练模型")
        model_was_loaded = True
    else:
        print(f"在 {MODEL_SAVE_PATH} 未找到模型文件。使用随机初始化的编码器进行测试。")
    loaded_base_encoder.to(DEVICE).eval()

    print("\n--- 运行Walker2d任务的相似性测试 ---")
    # 为测试任务定义并生成虚拟数据
    _, test_task_data = generate_mujoco_task_definitions_and_data(
        num_tasks=3,  # 未使用，下面手动定义
        timesteps_per_task=TIMESTEPS_PER_TASK,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        env_name="Walker2d"
    )
    test_task_defs = [
        {'name': 'Walker2d_target_vel_2.50', 'type': 'target_velocity', 'params': {'velocity': 2.5}},
        {'name': 'Walker2d_target_vel_2.60', 'type': 'target_velocity', 'params': {'velocity': 2.6}},  # 相似任务
        {'name': 'Walker2d_target_vel_0.10', 'type': 'target_velocity', 'params': {'velocity': 0.1}},  # 不相似任务
    ]

    print("正在为相似性测试任务生成嵌入...")
    test_embeddings = get_embeddings_for_inference(
        loaded_base_encoder, test_task_defs, test_task_data, N_SAMPLES_PER_VIEW, DEVICE
    )

    if test_embeddings.shape[0] == len(test_task_defs):
        emb_fast = test_embeddings[0]
        emb_fast_sim = test_embeddings[1]
        emb_slow = test_embeddings[2]

        def calculate_cosine_sim(v1, v2):
            return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

        sim_fast_fast = calculate_cosine_sim(emb_fast, emb_fast_sim)
        sim_fast_slow = calculate_cosine_sim(emb_fast, emb_slow)

        print("\n--- 余弦相似度 ---")
        print(f"Sim(跑得快, 跑得非常快): {sim_fast_fast:.4f} (期望: 高)")
        print(f"Sim(跑得快, 跑得慢): {sim_fast_slow:.4f} (期望: 低)")
    else:
        print("无法生成所有测试嵌入。")

    print("\n--- 测试完成 ---")
    if not model_was_loaded:
        print("注意: 相似度结果来自一个**随机初始化**的模型。")


if __name__ == '__main__':
    main_pretrain_and_test_walker()