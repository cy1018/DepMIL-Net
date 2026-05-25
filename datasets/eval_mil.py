# eval_mil.py
import torch


@torch.no_grad()
def eval_one_epoch(model, dataloader, device, task="regression"):
    """
    dataloader 输出:
      x: (B, S, N, C, t, H, W)  —— 因为Dataset返回(S,N,...)，batch后会多B维
      y: (B, ...) 标签
    task:
      - 'regression': 输出 shape 通常是 (B,1) 或 (B,)
      - 'classification': 输出 shape 通常是 (B,num_classes)
    """
    model.eval()
    preds = []
    gts = []

    for x, y in dataloader:
        x = x.to(device)  # (B,S,N,C,t,H,W)
        y = y.to(device)

        B, S, N, C, t, H, W = x.shape

        # 拉平 B*S 个bag，一次性 forward
        x_flat = x.view(B * S, N, C, t, H, W)

        out = model(x_flat)  # (B*S, D)  D=1 or num_classes

        # reshape 回来，对 S 段做平均
        out = out.view(B, S, -1).mean(dim=1)  # (B, D)

        preds.append(out.detach().cpu())
        gts.append(y.detach().cpu())

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)

    return preds, gts
