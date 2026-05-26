

from .M3DFEL import M3DFEL as M3DFEL_DFEW          # 原版（DFEW等分类）
from .DepMIL_Net import DepMIL_Net as DepMIL_AVEC     # AVEC版本（回归/多段）

def create_model(args):
   

    model_name = getattr(args, "model", "m3dfel_avec").lower()

    if model_name in ["DepMIL_avec", "avec", "avec2014"]:
        model = DepMIL_Net(args)
    elif model_name in ["m3dfel", "r3d", "m3dfel_r3d"]:
        model = M3DFEL_DFEW(args)
    else:
        raise ValueError(f"Unknown model: {args.model}. Use: m3dfel_avec or m3dfel")

    if args.num_frames % args.instance_length != 0:
        raise ValueError(
            f"num_frames({args.num_frames}) must be divisible by instance_length({args.instance_length})."
        )

 
    if getattr(args, "dataset", "").upper() in ["AVEC2014", "AVEC"] and getattr(args, "num_classes", None) != 1:
        print(f"[Warning] AVEC regression usually uses num_classes=1, but got num_classes={args.num_classes}.")

    return model
