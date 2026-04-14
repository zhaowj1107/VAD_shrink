# VAD 模型蒸馏训练方法

## 目录

1. [概述](#概述)
2. [知识蒸馏原理](#知识蒸馏原理)
3. [模型架构](#模型架构)
4. [训练配置](#训练配置)
5. [完整训练流程](#完整训练流程)
6. [代码结构](#代码结构)
7. [评估与验证](#评估与验证)
8. [常见问题](#常见问题)

---

## 概述

本文档描述了如何使用知识蒸馏（Knowledge Distillation）技术训练一个精简版的 VAD（Voice Activity Detection）模型。

### 目标

| 指标 | 原始 FP32 模型 | 蒸馏后目标 |
|------|----------------|------------|
| F1 分数 | 0.948 | ≥ 0.90 |
| 推理时间 | ~2.18s | < 1s |
| 参数量 | ~1M | < 0.5M |

### 数据集

使用 LibriParty 数据集进行训练和评估：
- **训练集**: 252 个 session
- **验证集**: 52 个 session
- **评估集**: 52 个 session

---

## 知识蒸馏原理

### 为什么需要知识蒸馏？

1. **模型压缩**: 将大模型的知识迁移到小模型
2. **保持精度**: 通过软标签学习 Teacher 的"暗知识"
3. **推理加速**: 小模型推理速度更快

### 蒸馏损失函数

使用 KL Divergence + BCE 混合损失：

```
L_total = α × L_KL + β × L_BCE
```

其中：
- **L_KL**: KL 散度损失，学习 Teacher 的软标签分布
- **L_BCE**: 二元交叉熵损失，学习真实标签
- **α = 0.7**, **β = 0.3** (默认权重)

### 温度缩放

使用温度参数 T = 2.0 来软化概率分布：

```
soft_prob = softmax(logits / T)
```

较高的温度会产生更平滑的概率分布，帮助 Student 学习类别间的相对关系。

---

## 模型架构

### Teacher 模型

- **来源**: SpeechBrain `vad-crdnn-libriparty`
- **精度**: FP32
- **输出**: 帧级语音概率 (10ms 网格)

### Student 模型 (Simplified CRDNN)

精简版 CRDNN 架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    Simplified CRDNN                        │
├─────────────────────────────────────────────────────────────┤
│  Input: (batch, time, 257) - Fbank features                │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐                        │
│  │  Conv2D    │    │  Conv2D     │    ← 2层 CNN           │
│  │  1→32 ch   │ →  │  32→64 ch   │    通道递减           │
│  │  3×3 kernel│    │  3×3 kernel │                        │
│  │  + ReLU    │    │  + ReLU     │                        │
│  │  + Pool    │    │  + Pool     │                        │
│  └─────────────┘    └─────────────┘                        │
│                        ↓                                    │
│  ┌─────────────────────────────────┐                       │
│  │  Bidirectional GRU              │    ← 1层 小型 GRU      │
│  │  hidden=128                     │                       │
│  └─────────────────────────────────┘                       │
│                        ↓                                    │
│  ┌─────────────────────────────────┐                       │
│  │  DNN (256 → 128) + ReLU + Dropout│    ← 1层 DNN        │
│  └─────────────────────────────────┘                       │
│                        ↓                                    │
│  ┌─────────────────────────────────┐                       │
│  │  Output (128 → 1) + Sigmoid     │    ← 语音概率        │
│  └─────────────────────────────────┘                       │
│                                                             │
│  Output: (batch, time) - 语音概率                          │
└─────────────────────────────────────────────────────────────┘
```

### 参数量对比

| 层级 | Teacher | Student | 压缩比 |
|------|---------|---------|--------|
| CNN | 3层 大通道 | 2层 小通道 | - |
| GRU | 2层 大隐藏 | 1层 小隐藏 | ~50% |
| DNN | 大隐藏层 | 小隐藏层 | ~50% |
| **总计** | ~1M | ~0.3M | **~70%** |

---

## 训练配置

### 默认超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `batch_size` | 8 | 批次大小 (GPU 8GB 够用) |
| `learning_rate` | 1e-3 | 学习率 |
| `max_epochs` | 20 | 最大训练轮数 |
| `early_stopping_patience` | 5 | 早停耐心值 |
| `gradient_clip` | 5.0 | 梯度裁剪阈值 |
| `kl_weight` | 0.7 | KL 损失权重 |
| `bce_weight` | 0.3 | BCE 损失权重 |
| `temperature` | 2.0 | 软标签温度 |

### 模型架构超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `cnn_channels` | (32, 64) | CNN 通道数 |
| `rnn_hidden_size` | 128 | GRU 隐藏单元 |
| `rnn_num_layers` | 1 | GRU 层数 |
| `dnn_hidden_size` | 128 | DNN 隐藏单元 |

---

## 完整训练流程

### 步骤 1: 安装依赖

```bash
pip install torch torchaudio speechbrain tqdm
```

### 步骤 2: 生成软标签

使用 FP32 Teacher 模型对训练集生成软标签：

```bash
PYTHONPATH=src python scripts/generate_soft_labels.py \
    --train-sessions-dir data/external/LibriParty/dataset/train \
    --output-dir data/processed/train_soft_labels
```

**输出**: `data/processed/train_soft_labels/*.npy` (每个 session 一个文件)

**预计时间**: 约 10-15 分钟 (252 个 session)

### 步骤 3: 训练 Student 模型

```bash
PYTHONPATH=src python scripts/train_student.py \
    --soft-labels-dir data/processed/train_soft_labels \
    --train-sessions-dir data/external/LibriParty/dataset/train \
    --dev-sessions-dir data/external/LibriParty/dataset/dev \
    --output-dir outputs/distillation \
    --epochs 20 \
    --batch-size 8 \
    --lr 0.001
```

**输出**:
- `outputs/distillation/checkpoints/epoch_*.pt` - 每轮检查点
- `outputs/distillation/checkpoints/best.pt` - 最佳模型

**预计时间**: 约 30-60 分钟 (取决于 GPU)

### 步骤 4: 评估模型

使用验证集评估模型性能：

```bash
PYTHONPATH=src python scripts/run_batch_evaluation.py \
    data/libriparty_dev_manifest.csv \
    --output-dir outputs/distilled_eval \
    --backend distilled \
    --checkpoint outputs/distillation/checkpoints/best.pt
```

### 自定义训练

#### 修改训练超参数

```bash
PYTHONPATH=src python scripts/train_student.py \
    --soft-labels-dir data/processed/train_soft_labels \
    --train-sessions-dir data/external/LibriParty/dataset/train \
    --dev-sessions-dir data/external/LibriParty/dataset/dev \
    --output-dir outputs/distillation_custom \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.0005
```

#### 快速测试 (使用验证集训练)

```bash
# 生成验证集软标签
PYTHONPATH=src python scripts/generate_soft_labels.py \
    --train-sessions-dir data/external/LibriParty/dataset/dev \
    --output-dir data/processed/dev_soft_labels

# 使用验证集训练和测试
PYTHONPATH=src python scripts/train_student.py \
    --soft-labels-dir data/processed/dev_soft_labels \
    --train-sessions-dir data/external/LibriParty/dataset/dev \
    --dev-sessions-dir data/external/LibriParty/dataset/dev \
    --output-dir outputs/distillation_test \
    --epochs 2 \
    --batch-size 2
```

---

## 代码结构

```
src/vad_baseline/distillation/
├── __init__.py              # 模块导出
├── config.py                # 训练配置 (DistillationConfig)
├── student_model.py          # SimplifiedCRDNN 模型定义
├── soft_label_generator.py  # 软标签生成器
├── trainer.py               # 训练器 (VADDistillationTrainer)
└── dataset.py               # 数据集加载

scripts/
├── generate_soft_labels.py   # 软标签生成入口
└── train_student.py         # 训练入口
```

### 核心类

#### DistillationConfig

训练配置数据类：

```python
from vad_baseline.distillation import DistillationConfig

config = DistillationConfig()
config.batch_size = 16
config.learning_rate = 0.0005
```

#### SimplifiedCRDNN

学生模型：

```python
from vad_baseline.distillation import SimplifiedCRDNN

model = SimplifiedCRDNN(
    input_size=257,
    cnn_channels=(32, 64),
    rnn_hidden_size=128,
    rnn_num_layers=1,
    dnn_hidden_size=128,
)
```

#### SoftLabelGenerator

软标签生成器：

```python
from vad_baseline.distillation import SoftLabelGenerator

generator = SoftLabelGenerator(
    teacher_model=vad_model,
    output_dir="data/processed/soft_labels",
)

generator.generate_for_sessions(["session_0", "session_1"])
```

#### VADDistillationTrainer

训练器：

```python
from vad_baseline.distillation import VADDistillationTrainer

trainer = VADDistillationTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    config=config,
)

trainer.train_epoch(train_loader)
dev_metrics = trainer.eval(dev_loader)
trainer.save_checkpoint("checkpoint.pt", epoch, dev_metrics["dev_f1"])
```

---

## 评估与验证

### 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| F1 | 精确率和召回率的调和平均 | ≥ 0.90 |
| Precision | 预测为语音中真正是语音的比例 | - |
| Recall | 真正语音被正确预测的比例 | - |
| Inference Time | 单个 session 推理时间 | < 1s |
| Parameters | 模型参数量 | < 0.5M |

### 评估脚本输出示例

```
Evaluating 52 sessions...
Session 1/52: session_0 - F1: 0.923
Session 2/52: session_1 - F1: 0.915
...

=== Summary ===
Mean F1: 0.918
Mean Precision: 0.925
Mean Recall: 0.911
Mean Inference Time: 0.82s
```

### 后端对比

训练完成后，可以使用 `DistilledBackend` 与其他后端进行对比：

```python
from vad_baseline.backends import get_backend

# 加载蒸馏模型
distilled = get_backend("distilled", checkpoint_path="outputs/distillation/checkpoints/best.pt")
distilled_model = distilled.load()

# 与其他后端对比
backends = ["distilled", "speechbrain_fp32", "webrtc_vad"]
for name in backends:
    backend = get_backend(name)
    model = backend.load()
    # 运行评估...
```

---

## 常见问题

### Q: 训练时间太长怎么办？

A: 减少训练 session 数量或 epochs：
```bash
# 使用子集训练
--train-sessions-dir data/external/LibriParty/dataset/dev  # 用验证集代替
--epochs 5
```

### Q: F1 分数低于目标怎么办？

A: 尝试以下方法：
1. 增加训练 epochs (到 30-50)
2. 调整学习率 (尝试 5e-4 或 5e-3)
3. 调整 kl_weight/bce_weight 比例
4. 增加模型容量 (更大的隐藏层)

### Q: GPU 内存不足怎么办？

A: 减小 batch size：
```bash
--batch-size 4  # 或 2
```

### Q: 如何继续训练？

A: 使用 `load_checkpoint` 方法：
```python
trainer = VADDistillationTrainer(...)
trainer.load_checkpoint("outputs/distillation/checkpoints/best.pt")
# 继续训练...
```

### Q: 软标签已经生成，如何重新训练？

A: 直接运行训练脚本，软标签会被复用：
```bash
PYTHONPATH=src python scripts/train_student.py \
    --soft-labels-dir data/processed/train_soft_labels \
    ...
```

---

## 参考

- Hinton et al. - "Distilling the Knowledge in a Neural Network" (2015)
- LibriParty Dataset: 多说话者混合语音数据集
- SpeechBrain VAD: `speechbrain/vad-crdnn-libriparty`
