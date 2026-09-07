# CS231n Assignments (2026) · Deep Learning for Computer Vision

Stanford CS231n 2026 三个编程项目的**完整实现与实验记录**（Assignment 1–3）

> Completed implementation & experiments for Stanford CS231n (2026) assignments. All code is written to be runnable locally; notebooks are localized (Colab Drive cells replaced with local setup) and ship with executed outputs.

---

## Overview · 项目简介

本项目是 CS231n 2026 课程三个 assignment 的独立完成版，包含全部待实现函数的代码、notebook 输出与 Inline Questions 解答：

| | 主题 | 覆盖内容 |
|---|---|---|
| **Assignment 1** | 图像分类与全连接网络 | kNN、Softmax/SVM、Two-Layer Net、Image Features、任意深度 FC Nets 与多种优化器 |
| **Assignment 2** | 卷积网络与 PyTorch | BatchNorm / LayerNorm / Dropout / ConvNet层手写实现、PyTorch 三层抽象实现CNN（裸张量 → nn.Module → nn.Sequential）、开放实现更好性能 |
| **Assignment 3** | 现代生成与多模态模型 | Transformer 图像描述与 ViT、SimCLR 自监督、DDPM 扩散生成、CLIP 零样本/检索与 DINO 分割 |

完成结果

| 任务 | 结果 |
|---|---|
| A1 kNN / Softmax / Two-Layer / FC 全链 | 全部达标（验证集 36% → 52% → 54.8% 等） |
| A2 PyTorch Part V（深层 CNN + 增强 + Adam） | 测试集 **83.6%** |
| A3 ViT（2 epoch CIFAR-10） | 测试 **45.6%** |
| A3 SimCLR 自监督（+线性评估） | 最佳 Top-1 **81.1%** |
| A3 DDPM | 完成采样与 Classifier-Free Guidance |
| A3 CLIP | 文本-图像相似度误差 ~1e-5，零样本分类与检索可用 |

---

## Repository Layout · 目录结构

```
.
├── README.md · requirements.txt
├── scripts/                        # 环境/数据准备脚本
│   ├── build_cifar10.py · set_kernelspec.py
│   └── localize_notebooks.py · fix_colab_paths.py
├── .envs/cs231n/                   # conda 环境（不在库中）
├── data/                           # 数据集缓存（不在库中）
│
└── assignments/
    ├── assignment1/                # 图像分类：kNN → Softmax → FC Nets → Features
    │   ├── knn.ipynb · softmax.ipynb
    │   ├── two_layer_net.ipynb · features.ipynb · FullyConnectedNets.ipynb
    │   └── cs231n/
    │       ├── classifiers/
    │       │   ├── k_nearest_neighbor.py      # 距离矩阵三种实现 + 投票
    │       │   ├── linear_classifier.py       # SGD 训练循环 + predict
    │       │   ├── softmax.py                 # naive / vectorized 损失与梯度
    │       │   └── fc_net.py                  # TwoLayerNet + FullyConnectedNet
    │       ├── layers.py · layer_utils.py     # affine/ReLU/softmax 等基础层
    │       ├── optim.py · solver.py           # SGD→Adam；Solver 训练器
    │       ├── features.py                    # HOG + HSV 直方图
    │       ├── data_utils.py · gradient_check.py · vis_utils.py
    │       └── datasets/ (本地数据，不入库)
    │
    ├── assignment2/                # 卷积网络：手写层 → PyTorch → RNN
    │   ├── BatchNormalization.ipynb · Dropout.ipynb
    │   ├── ConvolutionalNetworks.ipynb · PyTorch.ipynb
    │   ├── RNN_Captioning_pytorch.ipynb
    │   └── cs231n/
    │       ├── layers.py           # BN/LN/Dropout/Conv/Pool/Spatial-BN/GroupNorm
    │       ├── layer_utils.py · fast_layers.py · im2col.py
    │       ├── optim.py · solver.py
    │       ├── classifiers/
    │       │   ├── fc_net.py       # FullyConnectedNet（支持 norm/dropout）
    │       │   ├── cnn.py          # ThreeLayerConvNet
    │       │   └── rnn_pytorch.py  # CaptioningRNN
    │       ├── rnn_layers_pytorch.py
    │       ├── captioning_solver_pytorch.py · coco_utils.py
    │       └── datasets/ (本地数据，不入库)
    │
    └── assignment3/                #  Transformer / 自监督 / Diffusion / 多模态
        ├── Transformer_Captioning.ipynb
        ├── Self_Supervised_Learning.ipynb
        ├── DDPM.ipynb · CLIP_DINO.ipynb
        └── cs231n/
            ├── transformer_layers.py            # 多头注意力/位置编码/编码解码层
            ├── classifiers/transformer.py       # CaptioningTransformer + ViT
            ├── simclr/
            │   ├── contrastive_loss.py          # InfoNCE naive/vectorized
            │   ├── data_utils.py · model.py · utils.py
            ├── gaussian_diffusion.py · unet.py · ddpm_trainer.py
            ├── clip_dino.py                     # CLIP 工具 + DINO 分割
            ├── coco_utils.py · captioning_solver_transformer.py
            ├── emoji_dataset.py · image_utils.py
            └── datasets/ (本地数据/权重，不入库)
```

**Notebook ↔ 源码对照**：每个 notebook 的说明文字都会指明它调用哪个文件；需要实现的函数在源码中以 `TODO` 标注，本仓库已全部完成。`collect_submission.ipynb`（各 assignment 内）用于打包 Gradescope 提交物。

---

## Environment · 运行环境

- Python 3.11（conda，`./.envs/cs231n`）
- PyTorch 2.13 + CUDA 13.0（NVIDIA RTX 5060 实测可用）
- torchvision / clip / einops / h5py / opencv / scikit-learn 等（见 `requirements.txt`）
- Jupyter kernel `cs231n`（已注册，指向本地 E 盘环境）

运行方式：用 VS Code 打开本仓库根目录，打开任一 `*.ipynb`，选择内核 `Python 3.11 (cs231n)` 后按顺序执行即可。notebook 已包含执行输出，也可 `Kernel → Restart & Run All` 重新训练。

> 说明：数据集、预训练权重、conda 环境等大文件均不入库（见 `.gitignore`），本地按各 assignment 的 `cs231n/datasets` 与 `scripts/` 说明准备即可。

---

## Highlights · 学习要点（按 Assignment）

### Assignment 1 —— 用numpy实现深度学习
- kNN：三种距离实现体会向量化（13.9s → 31.5s → 0.09s 的实测对比）；
- Softmax：解析梯度推导 `p − onehot` 与数值梯度互验；
- Two-Layer / FC Nets：模块化 `forward/backward + cache`、链式法则、L2 正则（0.5 约定）、SGD → Momentum → RMSProp → Adam；
- Features：HOG + HSV 直方图，直观感受"特征工程 vs 原始像素"。

### Assignment 2 —— 现代网络的组件库
- BatchNorm / LayerNorm / Dropout / Conv / Pool / Spatial-BN / GroupNorm 全部手写并通过梯度检查；
- PyTorch 三层次抽象：同一网络分别用裸张量、`nn.Module`、`nn.Sequential` 实现，目标从 42% 提升到 55%（动量 + 更好初始化）；
- 开放挑战中通过"加深 + BatchNorm + 数据增强 + Adam"在 10 epoch 内把 CIFAR-10 验证集推到 80%+。

### Assignment 3 —— 现代生成与多模态
- Transformer：QKV 多头注意力、位置编码、Decoder/Encoder、ViT；
- SimCLR：InfoNCE 对比损失与自监督表征，线性评估 81%；
- DDPM：加噪/去噪、UNet、Classifier-Free Guidance；
- CLIP：图文共享空间中的相似度、零样本分类与图像检索；DINO patch 特征的可视化与分割。

---

## Notes · 环境与数据说明

- CIFAR-10 由 HuggingFace parquet 无损重建为官方 pickle 布局（数据源不可达时的本地方案），并修复了早期重建中通道布局的问题；
- 部分示例图片源（Flickr URL）已失效，相关 notebook 使用灰色占位图保证流程可运行；
- DINO 官方权重源不可达时，从 HuggingFace 镜像加载；DAVIS 真实数据需要 `tensorflow-datasets`，缺失时使用合成视频回退演示完整流程；
- 所有大文件与权重通过 `.gitignore` 排除，仓库仅保留代码与文档。

---

## Disclaimer · 声明

本项目为个人学习产出，仅用于学习与交流。请遵守斯坦福课程 [Honor Code](https://communitystandards.stanford.edu/policies-and-guidance/honor-code) 与生成式 AI 使用政策；建议读者在理解后独立完成作业。
