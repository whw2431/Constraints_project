#!/bin/bash

#SBATCH --job-name=simclr_gpu_pretrain    # 作业名称
#SBATCH --output=simclr_gpu_pretrain_%j.out # 标准输出日志 (%j 会被作业ID替换)
#SBATCH --error=simclr_gpu_pretrain_%j.err  # 标准错误日志

#SBATCH --partition=general_st
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1               # 每个节点运行1个任务
#SBATCH --cpus-per-task=4                 # 为每个任务分配的CPU核心数 (对应 DataLoader num_workers)
#SBATCH --mem=16G                         # 每个节点总内存 (或 --mem-per-cpu)
#SBATCH --time=02:00:00                   # 预计运行时间 (HH:MM:SS)

# --- DAIC 环境设置 ---
echo "Job started on $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR" # 通常是提交脚本的目录

# 清理并加载所需模块 (示例，请根据 DAIC 文档调整)
module purge
module load 2022r2 # 或者其他你使用的 DAIC toolchain
module load python/3.9.8 # 或者你需要的 Python 版本
module load cuda/11.7 # 或者与你 PyTorch 兼容的 CUDA 版本
module load cudnn/8.5.0.96-11.7 # CUDNN 版本
# module load anaconda3/xxxx # 如果你使用 Anaconda

# 激活你的 Conda 环境 (如果使用)
# CONDA_BASE=$(conda info --base)
# source $CONDA_BASE/etc/profile.d/conda.sh
# conda activate your_pytorch_env_name # 替换为你的环境名
# echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# 或者，如果使用 Python 虚拟环境 (venv)
# source /path/to/your/venv/bin/activate

# 检查 Python 和 PyTorch
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available to PyTorch: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"


# --- 运行 Python 脚本 ---
# 假设你的 Python 脚本名为 embedding_1D_adapted_gpu.py 并且和此 SLURM 脚本在同一目录
PYTHON_SCRIPT_NAME="embedding_1D_adapted_gpu.py"

 cd /tudelft.net/staff-umbrella/joerydevries/


echo "Running Python script for joerydevries pre-training..."
python embedding_1D_adapted_gpu.py

echo "Python script finished."
echo "Job finished."