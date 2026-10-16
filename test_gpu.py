"""第一个集群 GPU 测试脚本：验证 uv 环境 + GPU 分配是否正常。

在集群上由 job.slurm 调用运行：.venv/bin/python test_gpu.py
预期输出写进 作业号.out，核心是最后打出 torch.cuda.is_available() == True
"""

import socket
import sys


def main() -> int:
    print(f"所在节点: {socket.gethostname()}")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"可执行文件: {sys.executable}")  # 应该指向你项目目录下的 .venv

    try:
        import torch
    except ImportError:
        print("❌ 没找到 torch —— 检查 .venv 是否建好、是否用 .venv/bin/python 运行")
        return 1

    print(f"torch:    {torch.__version__}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")

    if not available:
        print("❌ CUDA 不可用 —— 大概率是作业没申请到 GPU（--gres=gpu:1）或队列不对")
        return 1

    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

    # 做一次真实的 GPU 运算，确认卡真的能算
    x = torch.randn(2048, 2048, device="cuda")
    y = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    z = x @ y
    torch.cuda.synchronize()
    print(f"GPU 矩阵乘法测试通过，结果形状: {tuple(z.shape)}, 采样值: {z[0, 0].item():.4f}")
    print("✅ 全部通过：uv 环境正常，GPU 已分配且可用。把 --partition 换成正式队列开始训练吧。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
