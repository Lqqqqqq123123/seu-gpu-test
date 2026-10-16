#!/bin/bash
# 第一个测试作业：先在 normal_test 队列验证流程，跑通后把 partition 改成 gpu_v100

#SBATCH --job-name=uv-gpu-test      # 作业名，不能有空格
#SBATCH --partition=normal_test     # 测试队列，30 分钟自动清场；正式训练改为 gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                # 申请 1 张卡
#SBATCH -o %j.out                   # 标准输出 -> 作业号.out
#SBATCH -e %j.err                   # 错误日志   -> 作业号.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xxxx@seu.edu.cn # 改成你的邮箱

# 用 uv 建的环境不需要 conda activate，直接用 .venv 里的 python
cd "$SLURM_SUBMIT_DIR"              # 进入你 sbatch 时所在的目录（即本文件所在目录）
export PATH="$HOME/.local/bin:$PATH"  # 保证能找到 uv（如果装过）

echo "== 环境 =="
which python || true
.venv/bin/python -V

echo "== GPU 测试 =="
.venv/bin/python test_gpu.py
