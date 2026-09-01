# CS231n 2026 本地作业环境（Stanford CS231n: Deep Learning for Computer Vision）

本目录是三个 assignment 的本地工作区，环境已配置完毕，开箱即可用。

## 目录结构

```
E:\ScaleLab\CS231n\project\
├── assignments\
│   ├── assignment1\   # Q1 kNN, Q2 Softmax, Q3 Two-Layer Net, Q4 Features, Q5 FC Nets
│   ├── assignment2\   # BatchNorm, Dropout, ConvNet, PyTorch-CIFAR10, RNN Captioning
│   └── assignment3\   # Transformer Captioning, Self-Supervised, DDPM, CLIP/DINO
├── data\              # 数据集缓存（CIFAR-10 重建产物、COCO zip、ImageNet val）
├── scripts\           # 环境/数据/notebook 本地化脚本
├── .envs\cs231n\      # conda 虚拟环境（Python 3.11 + PyTorch cu130），已 gitignore
├── requirements.txt   # 依赖清单（可重建环境）
└── README.md
```

## 环境概况

- Python 3.11（conda，位于 `.envs\cs231n`，全部在 E 盘）
- PyTorch 2.13.0 + CUDA 13.0（支持本机 RTX 5060，已实测 `torch.cuda.is_available() == True`）
- torchvision 0.28.0、OpenAI CLIP、Cython、opencv、decord、pyarrow、jupyter 等

## 快速开始

方式一（推荐，使用 Jupyter Notebook）：

```powershell
# 在项目根目录启动（注意：务必用 E 盘环境里的 python）
& "E:\ScaleLab\CS231n\project\.envs\cs231n\python.exe" -m jupyter notebook "E:\ScaleLab\CS231n\project\assignments"
```

方式二（JupyterLab）：

```powershell
& "E:\ScaleLab\CS231n\project\.envs\cs231n\python.exe" -m jupyter lab "E:\ScaleLab\CS231n\project\assignments"
```

打开任意作业的 notebook（已默认绑定 `Python 3.11 (cs231n)` 内核），从第一个单元格开始依次运行即可。

## 数据集

| 数据集 | 位置 | 说明 |
|---|---|---|
| CIFAR-10 | 每个作业的 `cs231n\datasets\cifar-10-batches-py`；assignment3 另有 `data\cifar-10-batches-py` | 已就绪 |
| COCO Captioning | assignment2/3 的 `cs231n\datasets\coco_captioning` | 需解压 `data\coco_captioning.zip`（脚本已处理） |
| ImageNet val (25张) | assignment2/3 的 `cs231n\datasets\imagenet_val_25.npz` | 已就绪 |

CIFAR-10 说明：官方下载源在国内极慢，本环境改为从 HuggingFace 镜像下载官方 parquet，
再用 `scripts\build_cifar10.py` 无损重建为标准 `cifar-10-batches-py` 格式（像素值与官方一致）。

## 对官方 notebook 做的本地化改动

1. 每个 notebook 第一个“挂载 Google Drive”的单元格已被替换为本地路径设置（设置 sys.path / 工作目录）。
2. 用 torchvision 的 notebook（PyTorch、Self-Supervised Learning、Transformer Captioning）
   注入了一个“本地数据补丁”单元格，让 torchvision 直接读取本地 CIFAR-10，避免触发慢速下载。
3. DDPM / CLIP_DINO 中写死 `/content/drive/...` 的输出路径已改为本地相对路径。
4. ConvolutionalNetworks 里的 Cython 编译单元已改为本地路径 + 当前解释器；
   若本机没有 MSVC 编译器导致编译失败，可跳过该单元，程序会自动使用纯 Python 的 im2col 实现。

原始未修改的 notebook 在 `E:\ScaleLab\CS231n\cs231n.github.io-master\assignments\2026\*.zip` 中，
如需恢复或将来在 Colab 上做最终提交，可从 zip 重新解压。

## 注意事项

- assignment3 的 CLIP / DINO 部分首次运行需要联网下载模型权重（OpenAI CLIP、facebookresearch/dino），
  之后会缓存在本机。
- `collect_submission.ipynb` 是提交用脚本，保留了 Colab 原版（生成 zip + PDF 并上传 Gradescope），
  本地运行不需要它。
- 训练较重的部分（如 Self-Supervised 完整训练、DDPM 5 万步）请按需调整参数，8GB 显存足够跑完本课程。
- 若想重建环境：

```powershell
conda create -p "E:\ScaleLab\CS231n\project\.envs\cs231n" python=3.11 -y --override-channels -c conda-forge
"E:\ScaleLab\CS231n\project\.envs\cs231n\python.exe" -m pip install -r requirements.txt
"E:\ScaleLab\CS231n\project\.envs\cs231n\python.exe" -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
"E:\ScaleLab\CS231n\project\.envs\cs231n\python.exe" -m pip install git+https://github.com/openai/CLIP.git
```

## 快速验证环境

```powershell
& "E:\ScaleLab\CS231n\project\.envs\cs231n\python.exe" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

