# 2026年2月25日，修改代码，就是修改损失函数
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import *
from utils import *

from datasets.main_dataloader_mil import AVECTrainMIL, AVECValTestMIL, build_avec_transform


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson_np(y_true, y_pred):
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


class Solver(object):
    def __init__(self, args):
        super(Solver, self).__init__()

        self.args = args
        self.log_path = os.path.join(self.args.output_path, "log.txt")

        # label z-score 统计（训练集）
        df = pd.read_csv(self.args.train_dataset)
        y = df["label"].astype(float).values
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() + 1e-8)

        # best records (dev)
        self.best_mae = 1e9
        self.best_rmse = 1e9
        self.best_pearson = -1e9

        # ✅ best history directory (lightweight)
        self.best_dir = os.path.join(self.args.output_path, "best_history")
        os.makedirs(self.best_dir, exist_ok=True)

        # ✅ keep last K best-history ckpts per metric (avoid disk explosion)
        self.keep_best_history = int(getattr(self.args, "keep_best_history", 10))

        if len(self.args.gpu_ids) > 0:
            torch.cuda.set_device(self.args.gpu_ids[0])
        self.device = torch.device(
            'cuda:%d' % self.args.gpu_ids[0] if self.args.gpu_ids else 'cpu'
        )

        seed = self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = create_model(self.args)
        if len(self.args.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, self.args.gpu_ids)
        self.model.to(self.device)

        if self.args.dataset != "AVEC2014":
            raise ValueError(
                "This solver_avec.py version is for AVEC2014 regression. "
                "If you want to run DFEW, use the original solver.py."
            )

        self.sample_interval = int(getattr(self.args, "sample_interval", 3))
        self.train_num_segs = int(getattr(self.args, "train_num_segs", 1))

        # ✅ val chunk size (avoid OOM)
        self.val_chunk_size = int(getattr(self.args, "val_chunk_size", 4))

        use_aug = bool(getattr(self.args, "random_sample", False))
        cj = float(getattr(self.args, "color_jitter", 0.4))
        train_transform = build_avec_transform(
            is_train=use_aug,
            color_jitter=cj,
            hflip_p=0.5
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            AVECTrainMIL(
                root=self.args.data_root,
                split="train",
                task=self.args.task,
                label_csv=self.args.train_dataset,
                frame_len=self.args.num_frames,
                bag_size=self.args.num_frames // self.args.instance_length,
                img_size=self.args.crop_size,
                sample_interval=self.sample_interval,
                input_channel=3,
                transform=train_transform,
                train_num_segs=self.train_num_segs
            ),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=False
        )

        if getattr(self.args, "dev_dataset", None) is None:
            raise ValueError("args.dev_dataset is None. Please set dev_label.csv in options.")

        self.val_dataloader = torch.utils.data.DataLoader(
            AVECValTestMIL(
                root=self.args.data_root,
                split="dev",
                task=self.args.task,
                label_csv=self.args.dev_dataset,
                frame_len=self.args.num_frames,
                bag_size=self.args.num_frames // self.args.instance_length,
                img_size=self.args.crop_size,
                sample_interval=self.sample_interval,
                input_channel=3,
                transform=None
            ),
            batch_size=1,
            shuffle=False,
            num_workers=max(1, self.args.workers // 2),
            pin_memory=True,
            drop_last=False
        )

        self.crit_l1 = nn.SmoothL1Loss().to(self.device)
        self.crit_mse = nn.MSELoss().to(self.device)
        self.w_l1 = float(getattr(self.args, "w_l1", 0.3))
        self.w_mse = float(getattr(self.args, "w_mse", 0.7))

        base_lr = float(self.args.lr)
        backbone_lr = base_lr * 0.1
        self.grad_clip = float(getattr(self.args, "grad_clip", 1.0))

        # accum steps
        self.accum_steps = int(getattr(self.args, "accum_steps", 4))

        backbone_params, head_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("features.") or name.startswith("module.features."):
                backbone_params.append(p)
            else:
                head_params.append(p)

        if len(backbone_params) > 0 and len(head_params) > 0:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": head_params, "lr": base_lr},
                ],
                eps=self.args.eps,
                weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=base_lr,
                eps=self.args.eps,
                weight_decay=self.args.weight_decay
            )

        self.scheduler = build_scheduler(self.args, self.optimizer, len(self.train_dataloader))

        if args.resume:
            # resume needs optimizer, so resume should load from model_latest.pth (full)
            checkpoint = torch.load(args.resume, map_location='cuda:0', weights_only=True)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            self.args.start_epoch = checkpoint['epoch'] + 1
            self.best_mae = checkpoint.get('best_mae', self.best_mae)
            self.best_rmse = checkpoint.get('best_rmse', self.best_rmse)
            self.best_pearson = checkpoint.get('best_pearson', self.best_pearson)
            self.model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _forward_video(self, images: torch.Tensor, is_train: bool = True) -> torch.Tensor:
        """
        train: (B, K, N, C, t, H, W) -> model -> mean over K
        val:   (B, S, N, C, t, H, W) -> chunk inference -> mean over S
        """
        if images.dim() == 6:
            return self.model(images)

        if images.dim() == 7:
            B, X, N, C, t, H, W = images.shape

            if is_train:
                x_flat = images.reshape(B * X, N, C, t, H, W)
                out = self.model(x_flat)
                out = out.view(B, X, -1).mean(dim=1)
                return out
            else:
                chunk = self.val_chunk_size
                out_chunks = []
                for start in range(0, X, chunk):
                    end = min(start + chunk, X)
                    x_chunk = images[:, start:end].reshape(B * (end - start), N, C, t, H, W)
                    with torch.no_grad():
                        o = self.model(x_chunk)
                    out_chunks.append(o.view(B, end - start, -1))
                out = torch.cat(out_chunks, dim=1).mean(dim=1)
                return out

        raise ValueError(f"Unexpected images dim={images.dim()}, shape={tuple(images.shape)}")

    def _scheduler_step(self, step_idx: int):
        if hasattr(self.scheduler, "step_update"):
            self.scheduler.step_update(step_idx)
        else:
            self.scheduler.step()

    def _norm_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def _denorm_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.y_std + self.y_mean

    def _loss(self, output_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
        loss_l1 = self.crit_l1(output_norm, target_norm)
        loss_mse = self.crit_mse(output_norm, target_norm)
        return self.w_l1 * loss_l1 + self.w_mse * loss_mse

    # ---------- save helpers (low disk usage) ----------
    def _state_light(self, state_full: dict) -> dict:
        """
        Lightweight checkpoint: used for best pointers and best_history.
        No optimizer to save disk.
        """
        return {
            "epoch": state_full["epoch"],
            "state_dict": state_full["state_dict"],
            "best_mae": state_full["best_mae"],
            "best_rmse": state_full["best_rmse"],
            "best_pearson": state_full["best_pearson"],
            "y_mean": state_full["y_mean"],
            "y_std": state_full["y_std"],
            "args": state_full["args"],  # you can comment out to save more space
        }

    def _prune_history(self, tag: str):
        if self.keep_best_history <= 0:
            return
        files = [f for f in os.listdir(self.best_dir)
                 if f.startswith(f"best_{tag}_epoch_") and f.endswith(".pth")]
        files.sort()  # filename contains epoch, lexicographic order works
        while len(files) > self.keep_best_history:
            rm = files.pop(0)
            try:
                os.remove(os.path.join(self.best_dir, rm))
            except Exception:
                break

    def run(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            inf = '********************' + str(epoch) + '********************'
            start_time = time.time()
            with open(self.log_path, 'a') as f:
                f.write(inf + '\n')
            print(inf)

            train_loss, train_metrics = self.train(epoch)
            val_loss, val_metrics = self.validate(epoch)

            # Determine best updates
            is_best_mae = val_metrics["mae"] < self.best_mae
            is_best_rmse = val_metrics["rmse"] < self.best_rmse
            is_best_pearson = val_metrics["pearson"] > self.best_pearson

            if is_best_mae:
                self.best_mae = val_metrics["mae"]
            if is_best_rmse:
                self.best_rmse = val_metrics["rmse"]
            if is_best_pearson:
                self.best_pearson = val_metrics["pearson"]

            # full state (latest uses this)
            state_full = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_mae': self.best_mae,
                'best_rmse': self.best_rmse,
                'best_pearson': self.best_pearson,
                'optimizer': self.optimizer.state_dict(),  # full only for latest/resume
                'args': self.args,
                'y_mean': self.y_mean,
                'y_std': self.y_std,
            }
            state_light = self._state_light(state_full)

            # save checkpoints (low disk)
            self.save(
                state_full=state_full,
                state_light=state_light,
                epoch=epoch,
                is_best_mae=is_best_mae,
                is_best_rmse=is_best_rmse,
                is_best_pearson=is_best_pearson
            )

            epoch_time = time.time() - start_time

            msg = (
                f"\nEpoch {epoch} Train : loss={train_loss:.4f}, "
                f"MAE={train_metrics['mae']:.4f}, RMSE={train_metrics['rmse']:.4f}, "
                f"Pearson={train_metrics['pearson']:.4f}\n"
                f"Epoch {epoch} Val   : loss={val_loss:.4f}, "
                f"MAE={val_metrics['mae']:.4f}, RMSE={val_metrics['rmse']:.4f}, "
                f"Pearson={val_metrics['pearson']:.4f}\n"
                f"Best so far         : MAE={self.best_mae:.4f}, RMSE={self.best_rmse:.4f}, "
                f"Pearson={self.best_pearson:.4f}\n"
                f"Epoch {epoch} Time  : {epoch_time:.1f}s\n\n"
            )
            with open(self.log_path, 'a') as f:
                f.write(msg)
            print(msg)

        return self.best_mae, self.best_rmse, self.best_pearson

    def train(self, epoch):
        self.model.train()

        all_pred, all_target = [], []
        all_loss = 0.0
        accum_steps = self.accum_steps
        self.optimizer.zero_grad(set_to_none=True)
        opt_step_in_epoch = 0

        for i, (images, target) in enumerate(self.train_dataloader):
            print(f"Training epoch {epoch}: {i + 1}/{len(self.train_dataloader)}", end='\r')

            images = images.to(self.device)
            target = target.to(self.device).float().view(-1, 1)
            target_norm = self._norm_y(target)

            output_norm = self._forward_video(images, is_train=True).view(-1, 1)
            loss = self._loss(output_norm, target_norm)

            (loss / accum_steps).backward()

            do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == len(self.train_dataloader))
            if do_step:
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                total_steps_per_epoch = (len(self.train_dataloader) + accum_steps - 1) // accum_steps
                step_idx = epoch * total_steps_per_epoch + opt_step_in_epoch
                self._scheduler_step(step_idx)
                opt_step_in_epoch += 1

            output = self._denorm_y(output_norm)
            all_loss += loss.item()
            all_pred.append(output.detach().cpu().numpy())
            all_target.append(target.detach().cpu().numpy())

        all_pred = np.concatenate(all_pred, axis=0).reshape(-1)
        all_target = np.concatenate(all_target, axis=0).reshape(-1)

        metrics = {
            "mae": mae_np(all_target, all_pred),
            "rmse": rmse_np(all_target, all_pred),
            "pearson": pearson_np(all_target, all_pred),
        }
        avg_loss = all_loss / max(1, len(self.train_dataloader))
        return avg_loss, metrics

    def validate(self, epoch):
        self.model.eval()

        all_pred, all_target = [], []
        all_loss = 0.0

        for i, (images, target) in enumerate(self.val_dataloader):
            print(f"Validating epoch {epoch}: {i + 1}/{len(self.val_dataloader)}", end='\r')

            images = images.to(self.device)
            target = target.to(self.device).float().view(-1, 1)
            target_norm = self._norm_y(target)

            with torch.no_grad():
                output_norm = self._forward_video(images, is_train=False).view(-1, 1)
                loss = self._loss(output_norm, target_norm)
                output = self._denorm_y(output_norm)

            all_loss += loss.item()
            all_pred.append(output.detach().cpu().numpy())
            all_target.append(target.detach().cpu().numpy())

        all_pred = np.concatenate(all_pred, axis=0).reshape(-1)
        all_target = np.concatenate(all_target, axis=0).reshape(-1)

        metrics = {
            "mae": mae_np(all_target, all_pred),
            "rmse": rmse_np(all_target, all_pred),
            "pearson": pearson_np(all_target, all_pred),
        }
        avg_loss = all_loss / max(1, len(self.val_dataloader))
        return avg_loss, metrics

    def save(self, state_full, state_light, epoch,
             is_best_mae=False, is_best_rmse=False, is_best_pearson=False):
        """
        Save checkpoints (disk-friendly):
          - always: model_latest.pth (FULL, with optimizer)
          - best pointers: best_mae.pth / best_rmse.pth / best_pearson.pth (LIGHT, overwrite)
          - best history: best_history/best_xxx_epoch_XXX_xxx_0.1234.pth (LIGHT, keep last K)
        """
        # 1) always latest (full)
        latest_path = os.path.join(self.args.output_path, "model_latest.pth")
        torch.save(state_full, latest_path)

        def _save_best(tag: str, metric_value: float, better: bool):
            if not better:
                return

            # pointer file (light, overwrite)
            ptr_path = os.path.join(self.args.output_path, f"best_{tag}.pth")
            torch.save(state_light, ptr_path)

            # history file (light, never overwrite)
            hist_name = f"best_{tag}_epoch_{epoch+1:03d}_{tag}_{metric_value:.4f}.pth"
            hist_path = os.path.join(self.best_dir, hist_name)
            torch.save(state_light, hist_path)

            # prune old history
            self._prune_history(tag)

            print(f"[BEST] best_{tag} updated -> ptr(light): {ptr_path}")
            print(f"[BEST] best_{tag} saved(light) -> hist: {hist_path}")

        _save_best("mae", float(state_full["best_mae"]), is_best_mae)
        _save_best("rmse", float(state_full["best_rmse"]), is_best_rmse)
        _save_best("pearson", float(state_full["best_pearson"]), is_best_pearson)


