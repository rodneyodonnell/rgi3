"""
Train functions, based on https://github.com/karpathy/nanoGPT/blob/master/train.py
"""

import dataclasses
import math
import time
import os
from collections import defaultdict

import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader

from rgi.rgizero import common as transformer_common


@dataclasses.dataclass
class TrainConfig:
    model_name: str
    model_version: str

    # Logging
    eval_interval: int = 2000
    log_interval: int = 100
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = True  # if True, always save a checkpoint after each eval

    # wandb logging
    wandb_log: bool = False  # disabled by default
    # wandb_project = "owt"
    # wandb_run_name = "gpt2"  # 'run' + str(time.time())

    # data
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    batch_size: int = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size

    # # adamw optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_epochs: int = 10  # maximum number of epochs (full passes through the dataset)
    max_iters: int = 5000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    # dtype: str = "float16"
    dtype: str = "bfloat16"
    compile: bool = False
    early_stop_patience: int = 5  # Early stopping patience


class Trainer:
    def __init__(
        self, model: nn.Module, train_config: TrainConfig, train_loader: DataLoader, val_loader: DataLoader, device: str
    ):
        self.model = model
        self.train_config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.scaler = torch.amp.GradScaler(enabled=(self.train_config.dtype == "float16"))
        self.optimizer = self.model.configure_optimizers(
            self.train_config.weight_decay,
            self.train_config.learning_rate,
            (self.train_config.beta1, self.train_config.beta2),
            self.device,
        )
        if self.train_config.compile:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)  # requires PyTorch 2.0
        # logging
        if train_config.wandb_log:
            import wandb

            wandb.init(
                project=self.train_config.wandb_project, name=self.train_config.wandb_run_name, config=self.model_config
            )
        self.ctx = transformer_common.get_ctx(self.train_config.dtype, self.device)
        self.model_dir = transformer_common.model_dir(self.train_config.model_name, self.train_config.model_version)
        os.makedirs(self.model_dir, exist_ok=True)
        self.no_improve_count = 0
        self.early_stop = False

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()  # put model in evaluation mode

        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            loss_sums = defaultdict(float)
            data_iter = iter(loader)
            eval_iters = min(self.train_config.eval_iters, len(loader))
            for k in range(eval_iters):
                data_batch = next(data_iter)
                with self.ctx:
                    logits, loss_dict, loss = self.model(*data_batch)
                loss_sums[split] += loss.item()
                for k, v in loss_dict.items():
                    loss_sums[split + "_" + k] += v.item()
                loss_average = {k: v / eval_iters for k, v in loss_sums.items()}
                out.update(loss_average)

        self.model.train()  # put model back into training mode
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.train_config.warmup_iters:
            return self.train_config.learning_rate * (it + 1) / (self.train_config.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.train_config.lr_decay_iters:
            return self.train_config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.train_config.warmup_iters) / (
            self.train_config.lr_decay_iters - self.train_config.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.train_config.min_lr + coeff * (self.train_config.learning_rate - self.train_config.min_lr)

    def train(self):
        for epoch_id in range(self.train_config.max_epochs):
            self.train_epoch(epoch_id)
            # termination conditions
            if self.iter_num > self.train_config.max_iters or self.early_stop:
                break

        # Reload best model if we saved one
        best_path = os.path.join(self.model_dir, "best.pt")
        if os.path.exists(best_path):
            print(f"Reloading best model from {best_path} (val_loss={self.best_val_loss:.4f})")
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])

    def train_epoch(self, epoch_id):
        self.model.train()
        assert torch.is_grad_enabled(), "Gradient should be enabled for training"
        data_iter = enumerate(self.train_loader)
        # Stop when the train_loader has been iterated over, or when we hit the global max_iters.
        max_iters = min(self.iter_num + len(self.train_loader), self.train_config.max_iters)
        t0 = time.time()
        t0_iter = self.iter_num
        while True:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.iter_num) if self.train_config.decay_lr else self.train_config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if (self.iter_num % self.train_config.eval_interval == 0) or (self.iter_num + 1 == max_iters):
                losses = self.estimate_loss()
                loss_str = ", ".join(f"{k}:{v:.4f}" for k, v in losses.items())
                print(f"step {self.iter_num}: losses: {loss_str}")
                if self.train_config.wandb_log:
                    wandb.log({"iter": self.iter_num, "lr": lr, **losses})
                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self.no_improve_count = 0

                    checkpoint = {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "model_args": "model_args",
                        "iter_num": self.iter_num,
                        "best_val_loss": self.best_val_loss,
                    }
                    print(f"saving best checkpoint to {self.model_dir}/best.pt")
                    torch.save(checkpoint, os.path.join(self.model_dir, "best.pt"))
                else:
                    self.no_improve_count += 1
                    if self.no_improve_count >= self.train_config.early_stop_patience:
                        print(
                            f"Early stopping triggered! Valid loss has not improved for {self.no_improve_count} evals."
                        )
                        self.early_stop = True
                        break

                if self.train_config.always_save_checkpoint:
                    if self.iter_num > 0:
                        checkpoint = {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "model_args": "model_args",  # TODO: Save arg dict.
                            "iter_num": self.iter_num,
                            "best_val_loss": self.best_val_loss,
                            # "config": self.model_config,  # TODO: save model_config
                        }
                        print(f"saving checkpoint to {self.model_dir}")
                        torch.save(checkpoint, os.path.join(self.model_dir, "ckpt.pt"))
            if self.iter_num == 0 and self.train_config.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.train_config.gradient_accumulation_steps):
                with self.ctx:
                    batch_id, data_batch = next(data_iter)
                    logits, loss_dict, loss = self.model(*data_batch)
                    loss = (
                        loss / self.train_config.gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()
            # clip the gradient
            if self.train_config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            if self.iter_num % self.train_config.log_interval == 0:
                # timing and logging
                t1 = time.time()
                t1_iter = self.iter_num
                dt = t1 - t0
                d_iter = t1_iter - t0_iter
                t0 = t1
                t0_iter = t1_iter
                ms_per_iter = (1_000 * dt / d_iter) if d_iter else 0
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                loss_str = ", ".join(
                    f"{k}:{v * self.train_config.gradient_accumulation_steps:.4f}" for k, v in loss_dict.items()
                )

                lossf = loss.item() * self.train_config.gradient_accumulation_steps
                print(
                    f"iter {self.iter_num}/{max_iters}/{self.train_config.max_iters}: loss {lossf:.4f}, {loss_str}, time {dt:.2f}s, iter_time: {ms_per_iter:.2f}ms"
                )
            self.iter_num += 1

            # termination conditions
            if self.iter_num >= max_iters:
                break
