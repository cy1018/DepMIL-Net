
import os
import argparse
import csv
import numpy as np
import torch

from datasets.main_dataloader_mil import AVECValTestMIL
from models import create_model


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


def forward_video(model, images: torch.Tensor) -> torch.Tensor:
    """
    images:
      - val/test dataset may output [B, S, N, C, t, H, W]
      - or [B, N, C, t, H, W]
    """
    if images.dim() == 6:
        return model(images)
    if images.dim() == 7:
        B, S, N, C, t, H, W = images.shape
        x = images.reshape(B * S, N, C, t, H, W)
        out = model(x)
        out = out.view(B, S, -1).mean(1)
        return out
    raise ValueError(f"Unexpected images shape: {tuple(images.shape)}")


def load_state_dict_safely(model, sd):
    try:
        model.load_state_dict(sd, strict=True)
        return
    except RuntimeError:
        pass

    model_keys = list(model.state_dict().keys())
    want_module = model_keys[0].startswith("module.")
    new_sd = {}
    for k, v in sd.items():
        if want_module and not k.startswith("module."):
            new_sd["module." + k] = v
        elif (not want_module) and k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)


def _clean_path(x):
    if x is None:
        return None
    return os.path.normpath(str(x).strip())


def load_checkpoint_anyway(path, device):
    """
    PyTorch 2.6+ safe loading:
    1) allowlist argparse.Namespace, then try weights_only=True
    2) fallback weights_only=False (trusted ckpt)
    """
    import torch.serialization
    torch.serialization.add_safe_globals([argparse.Namespace])

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as e1:
        print("[WARN] weights_only=True failed, fallback to weights_only=False")
        print("       reason:", repr(e1))
        return torch.load(path, map_location=device, weights_only=False)


def _stats(arr: np.ndarray):
    arr = arr.astype(np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _looks_like_zscore(y_pred_raw: np.ndarray) -> bool:
    """
    Loose heuristic: output resembles z-score space if values are not huge.
    """
    m = float(np.mean(y_pred_raw))
    s = float(np.std(y_pred_raw))
    mn = float(np.min(y_pred_raw))
    mx = float(np.max(y_pred_raw))

    if abs(m) < 3.0 and s < 3.0 and mn > -10.0 and mx < 10.0:
        return True
    return False


def _to_list(x):
    """Convert possible vid types to python list[str]."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dim() == 0:
            return [str(x.item())]
        return [str(v) for v in x.tolist()]
    # fallback single value
    return [str(x)]


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", type=str, default=r"E:\cyy_project\M3DFEL _1\avec14_processed")
    p.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    p.add_argument("--task", type=str, default="Freeform")
    p.add_argument("--label_csv", type=str, default=None)

    p.add_argument("--output_path", type=str,
                   default=r"E:\cyy_project\M3DFEL\outputs_avec\AVEC2014_Freeform_02-02-09-22")
    p.add_argument("--ckpt", type=str, default=None)

    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--instance_length", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--sample_interval", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)

    # ===== make test model match train config =====
    p.add_argument("--model", type=str, default="m3dfel_avec", help="model name used in training")
    p.add_argument("--use_dual_stream", action="store_true",
                   help="enable dual-stream forward during test (must match training)")
    p.add_argument("--score_tau", type=float, default=0.2,
                   help="temperature for soft selection (must match training if dual-stream)")

    # ===== PGFE flags =====
    p.add_argument("--use_pgfe", action="store_true",
                   help="enable PGFE for Stream-B (must match training if used)")
    p.add_argument("--pgfe_type", type=str, default="soft", choices=["soft", "inv"],
                   help="PGFE type: soft or inv")
    p.add_argument("--pgfe_gamma", type=float, default=0.2,
                   help="PGFE gamma for soft mode")
    p.add_argument("--proto_detach", action="store_true",
                   help="detach prototype when guiding Stream-B (recommended)")

    # Optional: train csv to compute denorm
    p.add_argument("--train_csv", type=str, default=None,
                   help="train_label.csv path to compute y_mean/y_std for denorm (optional)")

    # Debug denorm controls
    p.add_argument("--force_denorm", action="store_true",
                   help="force apply denorm (z*std+mean)")
    p.add_argument("--no_denorm", action="store_true",
                   help="force disable denorm")

    # NEW: save predictions
    p.add_argument("--save_pred_csv", action="store_true",
                   help="save per-sample predictions to CSV next to ckpt")
    p.add_argument("--pred_csv", type=str, default=None,
                   help="custom csv path for saving predictions (optional)")
    p.add_argument("--topk_err", type=int, default=20,
                   help="print top-k absolute error samples")

    args = p.parse_args()

    args.data_root = _clean_path(args.data_root)
    args.output_path = _clean_path(args.output_path)
    args.label_csv = _clean_path(args.label_csv)
    args.ckpt = _clean_path(args.ckpt)
    args.train_csv = _clean_path(args.train_csv)
    args.pred_csv = _clean_path(args.pred_csv)

    if not args.label_csv:
        default_csv = "dev_label.csv" if args.split == "dev" else "test_label.csv"
        args.label_csv = os.path.join(args.data_root, default_csv)

    if not args.ckpt:
        args.ckpt = os.path.join(args.output_path, "model_best.pth")

    print("[DEBUG] data_root =", repr(args.data_root))
    print("[DEBUG] label_csv =", repr(args.label_csv))
    print("[DEBUG] ckpt      =", repr(args.ckpt))
    print("[DEBUG] train_csv =", repr(args.train_csv))
    print("[DEBUG] model cfg :",
          f"model={args.model}, use_dual_stream={args.use_dual_stream}, score_tau={args.score_tau}, "
          f"use_pgfe={args.use_pgfe}, pgfe_type={args.pgfe_type}, pgfe_gamma={args.pgfe_gamma}, "
          f"proto_detach={args.proto_detach}")

    if not os.path.exists(args.label_csv):
        raise FileNotFoundError(f"label_csv not found: {args.label_csv}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt}")

    if args.use_pgfe and (not args.use_dual_stream):
        print("[WARN] --use_pgfe is enabled but --use_dual_stream is False. PGFE will have no effect.")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # load ckpt
    state = load_checkpoint_anyway(args.ckpt, device)

    # state_dict
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state

    # build model args (MUST match training)
    class DummyArgs:
        pass

    model_args = DummyArgs()
    model_args.dataset = "AVEC2014"
    model_args.model = args.model
    model_args.num_frames = args.num_frames
    model_args.instance_length = args.instance_length
    model_args.crop_size = args.crop_size
    model_args.gpu_ids = [args.gpu]
    model_args.num_classes = 1

    model_args.use_dual_stream = bool(args.use_dual_stream)
    model_args.score_tau = float(args.score_tau)

    model_args.use_pgfe = bool(args.use_pgfe)
    model_args.pgfe_type = str(args.pgfe_type)
    model_args.pgfe_gamma = float(args.pgfe_gamma)
    model_args.proto_detach = bool(args.proto_detach)

    model = create_model(model_args)
    model.to(device)
    model.eval()
    load_state_dict_safely(model, sd)

    # dataloader
    bag_size = args.num_frames // args.instance_length
    ds = AVECValTestMIL(
        root=args.data_root,
        split=args.split,
        task=args.task,
        label_csv=args.label_csv,
        frame_len=args.num_frames,
        bag_size=bag_size,
        img_size=args.crop_size,
        sample_interval=args.sample_interval,
        input_channel=3,
        transform=None,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=True,
        drop_last=False,
    )

    # inference (raw pred) + collect per-sample rows
    rows = []
    all_pred_raw, all_tgt = [], []
    global_idx = 0

    with torch.no_grad():
        for batch in loader:
            # support dataset returning (images, target) or (images, target, vid)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, target = batch
                vid_list = None
            elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
                images, target, vid = batch[0], batch[1], batch[2]
                vid_list = _to_list(vid)
            else:
                raise ValueError("Unexpected batch format from dataloader.")

            images = images.to(device)
            target = target.float().view(-1, 1).to(device)

            out = forward_video(model, images).view(-1, 1)  # raw
            pred_raw_np = out.detach().cpu().numpy().reshape(-1)
            tgt_np = target.detach().cpu().numpy().reshape(-1)

            all_pred_raw.append(pred_raw_np)
            all_tgt.append(tgt_np)

            bs = len(pred_raw_np)
            for i in range(bs):
                row = {
                    "idx": int(global_idx),
                    "vid": vid_list[i] if (vid_list is not None and i < len(vid_list)) else "",
                    "y_true": float(tgt_np[i]),
                    "y_pred_raw": float(pred_raw_np[i]),
                }
                rows.append(row)
                global_idx += 1

    y_pred_raw = np.concatenate(all_pred_raw, 0).reshape(-1)
    y_true = np.concatenate(all_tgt, 0).reshape(-1)

    # denorm params (prefer ckpt y_mean/y_std)
    have_denorm_params = False
    y_mean, y_std = 0.0, 1.0
    denorm_source = "none"

    if isinstance(state, dict) and ("y_mean" in state) and ("y_std" in state):
        y_mean = float(state["y_mean"])
        y_std = float(state["y_std"])
        have_denorm_params = True
        denorm_source = "ckpt(y_mean/y_std)"
    elif args.train_csv and os.path.exists(args.train_csv):
        import pandas as pd
        df = pd.read_csv(args.train_csv)
        y = df["label"].astype(float).values
        y_mean = float(y.mean())
        y_std = float(y.std() + 1e-8)
        have_denorm_params = True
        denorm_source = "train_csv(mean/std)"

    # denorm decision
    auto_need_denorm = _looks_like_zscore(y_pred_raw)
    if args.no_denorm:
        do_denorm = False
        reason = "--no_denorm"
    elif args.force_denorm:
        do_denorm = True
        reason = "--force_denorm"
    else:
        do_denorm = bool(auto_need_denorm and have_denorm_params)
        reason = f"auto(zscore={auto_need_denorm}, have_params={have_denorm_params})"

    if do_denorm:
        y_pred = y_pred_raw * y_std + y_mean
    else:
        y_pred = y_pred_raw

    # fill rows with denorm pred + errors
    for i, r in enumerate(rows):
        yp = float(y_pred[i])
        yt = float(r["y_true"])
        r["y_pred"] = yp
        r["err"] = float(yt - yp)
        r["abs_err"] = float(abs(yt - yp))

    print("[DEBUG] auto zscore check:", auto_need_denorm)
    print("[DEBUG] denorm params:", have_denorm_params, "source=", denorm_source,
          "y_mean=", y_mean, "y_std=", y_std)
    print("[DEBUG] denorm decision:", do_denorm, "reason=", reason)

    # metrics
    mae = mae_np(y_true, y_pred)
    rmse = rmse_np(y_true, y_pred)
    pr = pearson_np(y_true, y_pred)

    print("\n========== EVAL RESULT ==========")
    print(f"ckpt      : {args.ckpt}")
    print(f"data_root : {args.data_root}")
    print(f"split     : {args.split}")
    print(f"task      : {args.task}")
    print(f"label_csv : {args.label_csv}")
    print(f"model     : {args.model}")
    print(f"dual      : {args.use_dual_stream}  (score_tau={args.score_tau})")
    print(f"pgfe      : {args.use_pgfe}  (type={args.pgfe_type}, gamma={args.pgfe_gamma}, proto_detach={args.proto_detach})")
    print(f"denorm    : {do_denorm}  ({reason}, source={denorm_source})")
    print(f"MAE={mae:.4f}  RMSE={rmse:.4f}  Pearson={pr:.4f}")

    print("\n[STATS] y_true    :", _stats(y_true))
    print("[STATS] y_pred_raw:", _stats(y_pred_raw))
    print("[STATS] y_pred    :", _stats(y_pred))
    print("=================================\n")

    # save csv
    if args.save_pred_csv or args.pred_csv:
        if args.pred_csv:
            save_path = args.pred_csv
        else:
            ckpt_dir = os.path.dirname(args.ckpt)
            save_path = os.path.join(ckpt_dir, f"pred_{args.split}_{args.task}.csv")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["idx", "vid", "y_true", "y_pred_raw", "y_pred", "err", "abs_err"]
            )
            w.writeheader()
            w.writerows(rows)

        print(f"[SAVE] prediction csv -> {save_path}")

        # print top-k
        k = max(0, int(args.topk_err))
        if k > 0:
            top = sorted(rows, key=lambda r: r["abs_err"], reverse=True)[:k]
            print(f"\n[TOP-{k} ABS ERROR]")
            for r in top:
                print(f"idx={r['idx']:03d}  vid={r['vid']}  y_true={r['y_true']:.2f}  "
                      f"y_pred={r['y_pred']:.2f}  abs_err={r['abs_err']:.2f}")
            print("")

if __name__ == "__main__":
    main()

