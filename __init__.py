# from .M3DFEL import M3DFEL
#
#
# def create_model(args):
#     """create model according to args
#
#     Args:
#         args
#     """
#     model = M3DFEL(args)
#
#     return model


# models/__init__.py
# models/__init__.py

from .M3DFEL import M3DFEL as M3DFEL_DFEW          # 原版（DFEW等分类）
from .M3DFEL_avec import M3DFEL as M3DFEL_AVEC     # 你的AVEC版本（回归/多段）

def create_model(args):
    """
    根据 args.model 选择模型：
    - args.model = "m3dfel_avec"  -> 使用 M3DFEL_avec.py
    - args.model = "m3dfel"       -> 使用原版 M3DFEL.py
    """

    model_name = getattr(args, "model", "m3dfel_avec").lower()

    if model_name in ["m3dfel_avec", "avec", "avec2014"]:
        model = M3DFEL_AVEC(args)
    elif model_name in ["m3dfel", "r3d", "m3dfel_r3d"]:
        model = M3DFEL_DFEW(args)
    else:
        raise ValueError(f"Unknown model: {args.model}. Use: m3dfel_avec or m3dfel")

    # ✅ 防止配置错：num_frames 必须能整除 instance_length
    if args.num_frames % args.instance_length != 0:
        raise ValueError(
            f"num_frames({args.num_frames}) must be divisible by instance_length({args.instance_length})."
        )

    # ✅ 给AVEC任务一个友好提示（不强制）
    if getattr(args, "dataset", "").upper() in ["AVEC2014", "AVEC"] and getattr(args, "num_classes", None) != 1:
        print(f"[Warning] AVEC regression usually uses num_classes=1, but got num_classes={args.num_classes}.")

    return model
