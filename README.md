# 集群 GPU 测试项目（uv 版）

目标：照下面的命令走一遍，验证「代码上传 → uv 建环境 → sbatch 申请卡 → 程序在 GPU 上跑」整个流程。

文件说明：

- `test_gpu.py` —— 测试脚本，打印所在节点、torch/CUDA 状态，并做一次真实 GPU 运算
- `job.slurm` —— 作业提交脚本，告诉 Slurm 要 1 张卡、跑什么
- `README.md` —— 就是本文件，操作步骤

## 第 0 步：上传项目

把整个 `seu-gpu-test` 文件夹打成 zip，在智算平台「文件管理」里上传并解压。

## 第 1 步：建环境（远程终端，一次性）

打开智算平台 → 远程终端：

```bash
cd seu-gpu-test          # 进入解压后的项目目录（cd 到你解压的实际路径）

pip install uv           # 安装 uv（装到集群的家目录里，永久有效）
uv venv                  # 在当前目录创建 .venv
uv pip install torch     # 装 PyTorch（自带 CUDA 支持，不需要你操心驱动）
```

想严格复现本地环境的话，把本地的 `uv.lock` 一起上传，然后 `uv sync` 代替上面两行 install。

## 第 2 步：改作业脚本里的邮箱

编辑 `job.slurm`，把 `--mail-user=` 后面改成你的邮箱（不改也不影响运行，只是收不到通知）。

## 第 3 步：提交作业

```bash
sbatch job.slurm         # 返回一串数字 = 作业号
squeue -u $USER          # 看 PD(排队) / R(运行)
```

## 第 4 步：看结果

```bash
cat <作业号>.out         # 应看到 ✅ 全部通过
cat <作业号>.err         # 有报错先看这里
speek -f <作业号>        # 或在终端里盯实时输出
```

`.out` 里重点看三行：`所在节点`（应该是 gpuXX 之类的计算节点）、`torch.cuda.is_available(): True`、最后一句 ✅。

## 第 5 步：跑通之后

1. 把 `job.slurm` 里 `--partition=normal_test` 改成 `gpu_v100`（或 `gpuB`），就是正式训练了。
2. 把 `test_gpu.py` 换成你的训练脚本，启动命令同理。
3. 以后每次只需要改代码 → 上传 → `sbatch job.slurm`，环境不用重建。

## 常见报错

| 现象 | 原因和办法 |
|------|-----------|
| `.err` 里 No such file: .venv/bin/python | 环境没建在项目目录里，回第 1 步确认 cd 到了项目目录再 `uv venv` |
| `torch.cuda.is_available(): False` | 作业没申请到卡：确认 `--gres=gpu:1` 存在、队列是 GPU 队列 |
| 一直 PD 排队 | normal_test/gpu 队列人多，`squeue` 看原因；或等 |
| pip install 很慢/失败 | 换源：`uv pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple` |
