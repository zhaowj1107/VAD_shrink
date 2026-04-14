# VAD 模型蒸馏训练设计文档

**日期**: 2026-04-14
**目标**: 通过知识蒸馏训练一个精简版 CRDNN VAD 模型
**精度目标**: F1 ≥ 0.90 (接近原始 FP32 的 0.948)
**推理时间目标**: < 1s

---

## 1. 概述

本项目旨在使用知识蒸馏技术，将 speechbrain/vad-crdnn-libriparty FP32 模型(Teacher) 的知识迁移到一个更小更快的 Simplified CRDNN 模型(Student)。训练完成后，Student 模型应能在保持 F1 ≥ 0.90 的同时，将推理时间从 2.18s 降低到 1s 以内。

---

## 2. 数据集

- **数据集**: LibriParty
- **路径**: `data/external/LibriParty/dataset/`
- **train sessions**: 252 个 session
- **dev sessions**: 52 个 session
- **音频格式**: WAV, ~19MB/session, ~5-10 分钟/session
- **标注格式**: JSON per session, 包含 speaker_id → [start, stop] 片段

---

## 3. Teacher 模型

- **来源**: speechbrain/vad-crdnn-libriparty (FP32 后端)
- **输出**: frame-level speech probabilities (10ms grid)
- **现有接口**: `inference.get_frame_probabilities_for_file()`

---

## 4. Student 模型设计 (Simplified CRDNN)

### 4.1 架构

| 层级 | 原始 CRDNN | Student (精简版) |
|------|------------|------------------|
| CNN | 3层, 大通道数 | 2层, 通道减半 |
| RNN | 2层 GRU, 大隐藏单元 | 1层 GRU, 小隐藏单元 |
| DNN | 大隐藏层 | 小隐藏层 |

### 4.2 目标规格

| 指标 | 目标 |
|------|------|
| 参数量 | ~0.3-0.5M (原 ~1M) |
| 推理时间 | < 1s per session |
| F1 分数 | ≥ 0.90 |

### 4.3 模型定义位置

`src/vad_baseline/distillation/student_model.py`

---

## 5. 训练流程

### Step 1: 生成软标签 (离线, 一次性)

```
for each train session:
    audio → FP32 Teacher → frame_probs (numpy)
    保存到: data/processed/train_soft_labels/{session_id}.npy
```

**脚本**: `scripts/generate_soft_labels.py`

### Step 2: 训练 Student

```
for epoch in range(max_epochs):
    for each train session:
        audio → Student → frame_probs
        loss = 0.7 * KL_div(Student_probs, Teacher_probs) + 0.3 * BCE(Student_probs, hard_labels)
        backward + optimizer.step()
    # eval on dev
    if dev_f1 > best_dev_f1:
        save best model
```

**脚本**: `scripts/train_student.py`

### Step 3: 评估

复用现有批量评估管道:
```bash
PYTHONPATH=src python scripts/run_batch_evaluation.py \
    data/libriparty_dev_manifest.csv \
    --output-dir outputs/distilled_eval \
    --backend distilled
```

---

## 6. 训练配置

| 超参 | 数值 |
|------|------|
| batch size | 8 (GPU 8GB 够用) |
| learning rate | 1e-3 |
| max epochs | 20 |
| early stopping | patience=5 on dev F1 |
| loss | 0.7 × KL_div + 0.3 × BCE |
| 软标签温度 T | 2.0 |
| optimizer | Adam |
| gradient clip | 5.0 |

---

## 7. 文件结构

```
src/vad_baseline/
    └── distillation/
        ├── __init__.py
        ├── student_model.py          # Simplified CRDNN 模型定义
        ├── soft_label_generator.py   # Step 1: 生成软标签逻辑
        ├── trainer.py                 # Step 2: 训练器
        └── config.py                  # 训练配置

scripts/
    ├── generate_soft_labels.py       # Step 1 入口
    └── train_student.py             # Step 2 入口

data/
    └── processed/                    # 新增
        └── train_soft_labels/       # Teacher 生成的软标签

outputs/
    └── distillation/                 # 新增
        ├── checkpoints/             # 模型检查点
        └── logs/                   # 训练日志
```

---

## 8. 后端集成

训练完成后，将 Student 模型集成到现有后端系统:

1. 创建 `src/vad_baseline/backends/distilled.py`
2. 实现 `DistilledBackend` 类 (继承 `BaseVADBackend`)
3. 注册到 `BACKEND_FACTORIES`

这样可以复用现有评估管道:
```bash
PYTHONPATH=src python scripts/run_batch_evaluation.py \
    <manifest.csv> \
    --backend distilled \
    --checkpoint outputs/distillation/checkpoints/best.pt
```

---

## 9. 验收标准

| 指标 | 目标 | 验证方法 |
|------|------|----------|
| F1 (dev) | ≥ 0.90 | 现有 metrics.py |
| 推理时间 | < 1s per session | benchmark.py |
| 参数量 | < 0.5M | model summary |

---

## 10. 依赖

- PyTorch 2.10.0+cpu (训练用 CPU 版即可, GPU 加速训练)
- speechbrain
- torchaudio
- numpy
- tqdm (进度条)

新增依赖会添加到 `requirements.txt` 的新 section。
