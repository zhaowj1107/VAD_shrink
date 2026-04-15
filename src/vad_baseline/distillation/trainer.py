# src/vad_baseline/distillation/trainer.py
"""Training logic for VAD distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def kl_divergence_loss(student_logits, teacher_logits, T=2.0):
    """
    Compute KL divergence loss between student and teacher outputs.

    Args:
        student_logits: Unbounded logits from student model
        teacher_logits: Unbounded logits from teacher model
        T: Temperature for softening distributions

    Returns:
        KL divergence loss (scalar)
    """
    # Apply temperature scaling
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)

    # KL divergence: sum(teacher * log(teacher/student)) = -sum(teacher * log(student/teacher))
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
    return kl * (T * T)  # Scale by T^2 as per Hinton et al.


def bce_loss(predictions, targets):
    """
    Compute BCE loss on hard labels.

    Args:
        predictions: Probabilities [0, 1]
        targets: Binary labels {0, 1}

    Returns:
        BCE loss (scalar)
    """
    return F.binary_cross_entropy(predictions, targets)


def distillation_loss(
    student_probs,
    teacher_probs,
    hard_labels,
    kl_weight=0.7,
    bce_weight=0.3,
    temperature=2.0,
):
    """
    Combined distillation loss: KL divergence + BCE.

    Args:
        student_probs: Student frame probabilities (after sigmoid)
        teacher_probs: Teacher frame probabilities
        hard_labels: Ground truth binary labels
        kl_weight: Weight for KL loss
        bce_weight: Weight for BCE loss
        temperature: Soft label temperature

    Returns:
        (total_loss, loss_dict) where loss_dict contains individual components
    """
    # Convert probabilities to logits for KL divergence
    eps = 1e-8
    student_logits = torch.log(student_probs.clamp(min=eps, max=1 - eps))
    teacher_logits = torch.log(teacher_probs.clamp(min=eps, max=1 - eps))

    kl = kl_divergence_loss(student_logits, teacher_logits, T=temperature)
    bce = bce_loss(student_probs, hard_labels)

    total = kl_weight * kl + bce_weight * bce

    loss_dict = {
        "kl": kl.item(),
        "bce": bce.item(),
        "total": total.item(),
    }

    return total, loss_dict


class VADDistillationTrainer:
    """Trainer for VAD distillation."""

    def __init__(
        self,
        student_model,
        teacher_model,
        config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.config = config
        self.device = device

        # Teacher in eval mode, no grad
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Optimizer for student
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=config.learning_rate,
        )

        self.best_dev_f1 = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        """Train one epoch."""
        self.student.train()

        total_loss = 0.0
        total_kl = 0.0
        total_bce = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            fbank = batch["fbank"].to(self.device)
            teacher_probs = batch["teacher_probs"].to(self.device)
            hard_labels = batch["hard_labels"].to(self.device)

            self.optimizer.zero_grad()

            # Student forward
            student_probs = self.student(fbank)

            # Compute loss
            loss, loss_dict = distillation_loss(
                student_probs=student_probs,
                teacher_probs=teacher_probs,
                hard_labels=hard_labels,
                kl_weight=self.config.kl_weight,
                bce_weight=self.config.bce_weight,
                temperature=self.config.temperature,
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config.gradient_clip,
            )
            self.optimizer.step()

            total_loss += loss_dict["total"]
            total_kl += loss_dict["kl"]
            total_bce += loss_dict["bce"]
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.4f}",
                "kl": f"{loss_dict['kl']:.4f}",
                "bce": f"{loss_dict['bce']:.4f}",
            })

        return {
            "train_loss": total_loss / num_batches,
            "train_kl": total_kl / num_batches,
            "train_bce": total_bce / num_batches,
        }

    def eval(self, dev_loader):
        """Evaluate on dev set."""
        self.student.eval()

        total_tp = 0
        total_fp = 0
        total_fn = 0

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating"):
                fbank = batch["fbank"].to(self.device)
                hard_labels = batch["hard_labels"].to(self.device)

                student_probs = self.student(fbank)

                # Compute metrics per sample (handle varying lengths)
                batch_size = student_probs.size(0)
                for i in range(batch_size):
                    probs = student_probs[i]
                    labels = hard_labels[i]

                    # Only consider non-zero elements (actual data, not padding)
                    valid_len = probs.size(0)
                    probs = probs[:valid_len]
                    labels = labels[:valid_len]

                    pred_binary = (probs > 0.5).float()
                    tp = (pred_binary * labels).sum().item()
                    fp = (pred_binary * (1 - labels)).sum().item()
                    fn = ((1 - pred_binary) * labels).sum().item()

                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {"dev_f1": f1, "dev_precision": precision, "dev_recall": recall}

    def save_checkpoint(self, path, epoch, dev_f1):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "student_state": self.student.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "dev_f1": dev_f1,
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint["student_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint.get("epoch", 0), checkpoint.get("dev_f1", 0.0)
