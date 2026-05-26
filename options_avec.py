#最终的完整代码
# import argparse
# import os
# import time


# class Options(object):

#     def __init__(self):
#         super(Options, self).__init__()

#     def initialize(self):
#         parser = argparse.ArgumentParser()

#         # ----------------------
#         # basic settings
#         # ----------------------
#         parser.add_argument('--mode', type=str, default="train")

#         # ✅ 默认直接跑 AVEC2014
#         parser.add_argument('--dataset', type=str, default="AVEC2014")

#         parser.add_argument('--gpu_ids', type=str, default='0',
#                             help='gpu ids, eg. 0,1,2; -1 for cpu.')
#         parser.add_argument('--resume', default=None, type=str,
#                             metavar='PATH', help='path to latest checkpoint')
#         parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
#                             help='manual epoch number (useful on restarts)')
#         parser.add_argument('--fold', default='1', type=str)   # AVEC 不用 fold，但留着不影响
#         parser.add_argument('--seed', default=42, type=int)

#         # ✅ 数据根目录
#         parser.add_argument('--data_root', default='avec14_processed', type=str)

#         # ✅ Freeform / Northwind
#         parser.add_argument('--task', default='Freeform', type=str)

#         # ✅ 新增：保存输出的根目录（你就用它来换保存位置）
#         parser.add_argument('--output_root', default='outputs', type=str,
#                             help='root directory to save logs/checkpoints')

#         # ----------------------
#         # numeric settings
#         # ----------------------
#         parser.add_argument('--workers', default=8, type=int,
#                             metavar='N', help='number of data loading workers')
#         parser.add_argument('--epochs', default=200, type=int,
#                             metavar='N', help='number of total epochs to run')
#         parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N')

#         # ✅ 回归默认 1（预测 BDI 分数）
#         parser.add_argument('--num_classes', default=1, type=int)

#         # ----------------------
#         # model settings (方案A)
#         # ----------------------
#         parser.add_argument('--num_frames', default=64, type=int, help='number of frames (per bag)')
#         parser.add_argument('--instance_length', default=4, type=int, metavar='N', help='instance length')
#         parser.add_argument('--crop_size', default=224, type=int, metavar='N', help='crop size (input size)')

#         parser.add_argument('--model', default='m3dfel_avec', type=str, help='Backbone')

#         # ----------------------
#         # ✅ Dual-stream MIL
#         # ----------------------
#         parser.add_argument('--use_dual_stream', action='store_true',
#                             help='enable dual-stream MIL (prototype-anchored key/context aggregation)')
#         parser.add_argument('--no_dual_stream', action='store_false', dest='use_dual_stream',
#                             help='disable dual-stream MIL')
#         parser.set_defaults(use_dual_stream=False)

#         # 兼容保留：hard top-k 版本会用；soft-selection 版本暂不使用该参数（不影响保留）
#         parser.add_argument('--topk', default=4, type=int,
#                             help='top-k instances for prototype anchor (hard top-k mode). '
#                                  'In soft-selection mode, this is not used.')

#         # ✅ 新增：soft selection 温度（你的修复版 dual-stream 会用）
#         parser.add_argument('--score_tau', default=0.2, type=float,
#                             help='temperature for soft instance selection in Stream-A '
#                                  '(smaller => sparser weights; typical: 0.1~0.5)')

#         # ----------------------
#         # ✅ PGFE (Prototype Distance-Guided Feature Enhancement) - Stream-B only
#         # ----------------------
#         parser.add_argument('--use_pgfe', action='store_true',
#                             help='enable PGFE (distance-guided enhancement) for Stream-B only')
#         parser.add_argument('--no_pgfe', action='store_false', dest='use_pgfe',
#                             help='disable PGFE')
#         parser.set_defaults(use_pgfe=False)

#         parser.add_argument('--pgfe_type', type=str, default='soft', choices=['soft', 'inv'],
#                             help='PGFE weighting: soft (recommended) or inv (1/d)')

#         parser.add_argument('--pgfe_gamma', type=float, default=0.2,
#                             help='smoothing factor for PGFE soft mode (larger -> smoother). typical: 0.1~0.5')

#         # ✅ 建议：默认 detach prototype，训练更稳
#         parser.add_argument('--proto_detach', action='store_true',
#                             help='detach prototype when guiding Stream-B (recommended for stability)')
#         parser.add_argument('--no_proto_detach', action='store_false', dest='proto_detach',
#                             help='do not detach prototype for Stream-B')
#         parser.set_defaults(proto_detach=True)

#         # ----------------------
#         # training hyperparameters
#         # ----------------------
#         parser.add_argument('--label_smoothing', default=0.0, type=float,
#                             help='label smoothing (regression usually 0)')

#         # ----------------------
#         # ✅ 新增：训练每个样本抽 K 个窗口
#         # ----------------------
#         parser.add_argument('--train_num_segs', default=1, type=int,
#                             help='number of random segments/windows sampled per video in training (K). '
#                                  'Set to 2/4 to increase training signal.')

#         # ----------------------
#         # ✅ sample_interval 可配（默认 3）
#         # ----------------------
#         parser.add_argument('--sample_interval', default=3, type=int,
#                             help='frame sampling interval (e.g., 1/2/3). Smaller keeps more micro-dynamics.')

#         # augmentation
#         parser.add_argument('--random_sample', action='store_true',
#                             help='enable random sampling/augmentation (if wired in dataloader)')
#         parser.add_argument('--no_random_sample', action='store_false', dest='random_sample',
#                             help='disable random sampling/augmentation')
#         parser.set_defaults(random_sample=False)

#         parser.add_argument('--color_jitter', default=0.4, type=float)

#         # optimizer
#         parser.add_argument('-o', '--optimizer', default="AdamW", type=str, metavar='Opti')

#         # ✅ 回归 + 预训练微调：更稳的起步 lr / wd
#         parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', dest='lr')
#         parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
#         parser.add_argument('--wd', '--weight_decay', default=1e-2, type=float, metavar='W', dest='weight_decay')
#         parser.add_argument('--eps', default=1e-8, type=float, metavar='EPSILON',
#                             help='Optimizer Epsilon (default: 1e-8)')

#         # ✅ 新增：梯度裁剪（配合 solver_avec.py）
#         parser.add_argument('--grad_clip', default=1.0, type=float,
#                             help='max grad norm; set 0 to disable')

#         # scheduler
#         parser.add_argument('--lr_scheduler', default="cosine", type=str)
#         parser.add_argument('--warmup_epochs', default=5, type=int)
#         parser.add_argument('--min_lr', default=5e-6, type=float)
#         parser.add_argument('--warmup_lr', default=0, type=float)

#         return parser

#     def parse(self):
#         parser = self.initialize()
#         args = parser.parse_args()

#         # ----------------------
#         # parse gpu ids
#         # ----------------------
#         str_ids = args.gpu_ids.split(',')
#         args.gpu_ids = []
#         for str_id in str_ids:
#             cur_id = int(str_id)
#             if cur_id >= 0:
#                 args.gpu_ids.append(cur_id)

#         # ----------------------
#         # output dir  ✅ 用 output_root
#         # ----------------------
#         time_str = time.strftime("%m-%d-%H-%M")
#         args.name = args.dataset + "_" + args.task + "_" + time_str

#         args.output_path = os.path.join(args.output_root, args.name)
#         os.makedirs(args.output_path, exist_ok=True)

#         # ----------------------
#         # dataset csv paths
#         # ----------------------
#         args.dev_dataset = None

#         if args.dataset == "AVEC2014":
#             args.train_dataset = os.path.join(args.data_root, "train_label.csv")
#             args.dev_dataset = os.path.join(args.data_root, "dev_label.csv")
#             args.test_dataset = os.path.join(args.data_root, "test_label.csv")

#             for p in [args.train_dataset, args.dev_dataset, args.test_dataset]:
#                 if not os.path.exists(p):
#                     raise FileNotFoundError(f"CSV not found: {p}")

#         elif args.dataset == "DFEW":
#             args.train_dataset = os.path.join(
#                 args.data_root, "EmoLabel_DataSplit/train(single-labeled)/set_X.csv")
#             args.test_dataset = os.path.join(
#                 args.data_root, "EmoLabel_DataSplit/test(single-labeled)/set_X.csv")
#             args.train_dataset = args.train_dataset.replace('X', str(args.fold))
#             args.test_dataset = args.test_dataset.replace('X', str(args.fold))

#         elif args.dataset == "FERV39K":
#             args.train_dataset = os.path.join(
#                 args.data_root, "FERV39K/FERV39k/4_setups/All_scenes/train_All.csv")
#             args.test_dataset = os.path.join(
#                 args.data_root, "FERV39K/FERV39k/4_setups/All_scenes/test_All.csv")

#         else:
#             raise ValueError(f"Unknown dataset: {args.dataset}")

#         # ----------------------
#         # safety checks
#         # ----------------------
#         if args.train_num_segs < 1:
#             raise ValueError(f"--train_num_segs must be >= 1, got {args.train_num_segs}")
#         if args.sample_interval < 1:
#             raise ValueError(f"--sample_interval must be >= 1, got {args.sample_interval}")

#         if args.topk < 1:
#             raise ValueError(f"--topk must be >= 1, got {args.topk}")

#         if args.score_tau <= 0:
#             raise ValueError(f"--score_tau must be > 0, got {args.score_tau}")

#         # PGFE checks
#         if args.use_pgfe:
#             if args.pgfe_gamma <= 0:
#                 raise ValueError(f"--pgfe_gamma must be > 0, got {args.pgfe_gamma}")
#             if not args.use_dual_stream:
#                 print("[Warn] --use_pgfe is enabled but --use_dual_stream is False. "
#                       "PGFE will have no effect (because prototype is not used).")

#         if args.instance_length < 1:
#             raise ValueError(f"--instance_length must be >= 1, got {args.instance_length}")
#         if args.num_frames % args.instance_length != 0:
#             raise ValueError(
#                 f"--num_frames ({args.num_frames}) must be divisible by --instance_length ({args.instance_length}) "
#                 f"because bag_size = num_frames//instance_length is used by pwconv."
#             )

#         bag_size = args.num_frames // args.instance_length
#         if args.topk > bag_size:
#             print(f"[Warn] --topk ({args.topk}) > bag_size ({bag_size}). Will clamp topk to {bag_size} in model.")
#         args.bag_size = bag_size

#         # info hints
#         if args.use_dual_stream:
#             print(f"[INFO] Dual-stream enabled. Using soft selection with score_tau={args.score_tau}. "
#                   f"(Note: --topk={args.topk} is not used in soft-selection mode.)")

#         if args.use_dual_stream and args.use_pgfe:
#             print(f"[INFO] PGFE enabled for Stream-B. type={args.pgfe_type}, gamma={args.pgfe_gamma}, "
#                   f"proto_detach={args.proto_detach}")

#         return args

#最终的代码完成


#为了消融实验修改的代码
import argparse
import os
import time


class Options(object):

    def __init__(self):
        super(Options, self).__init__()

    def initialize(self):
        parser = argparse.ArgumentParser()

        # ----------------------
        # basic settings
        # ----------------------
        parser.add_argument('--mode', type=str, default="train")

        # 默认直接跑 AVEC2014
        parser.add_argument('--dataset', type=str, default="AVEC2014")

        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids, eg. 0,1,2; -1 for cpu.')
        parser.add_argument('--resume', default=None, type=str,
                            metavar='PATH', help='path to latest checkpoint')
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--fold', default='1', type=str)   # AVEC 不用 fold，但留着不影响
        parser.add_argument('--seed', default=42, type=int)

        # 数据根目录
        parser.add_argument('--data_root', default='avec14_processed', type=str)

        # Freeform / Northwind
        parser.add_argument('--task', default='Freeform', type=str)

        # 保存输出的根目录
        parser.add_argument('--output_root', default='outputs', type=str,
                            help='root directory to save logs/checkpoints')

        # ----------------------
        # numeric settings
        # ----------------------
        parser.add_argument('--workers', default=8, type=int,
                            metavar='N', help='number of data loading workers')
        parser.add_argument('--epochs', default=200, type=int,
                            metavar='N', help='number of total epochs to run')
        parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N')

        # 回归默认 1（预测 BDI 分数）
        parser.add_argument('--num_classes', default=1, type=int)

        # ----------------------
        # model settings
        # ----------------------
        parser.add_argument('--num_frames', default=64, type=int, help='number of frames (per bag)')
        parser.add_argument('--instance_length', default=4, type=int, metavar='N', help='instance length')
        parser.add_argument('--crop_size', default=224, type=int, metavar='N', help='crop size (input size)')

        parser.add_argument('--model', default='m3dfel_avec', type=str, help='Backbone')

        # ----------------------
        # Dual-stream MIL
        # ----------------------
        parser.add_argument('--use_dual_stream', action='store_true',
                            help='enable dual-stream MIL (prototype-anchored key/context aggregation)')
        parser.add_argument('--no_dual_stream', action='store_false', dest='use_dual_stream',
                            help='disable dual-stream MIL')
        parser.set_defaults(use_dual_stream=False)

        # ===== 新增：Key / Context 分支开关 =====
        parser.add_argument('--use_key_branch', type=int, default=1, choices=[0, 1],
                            help='whether to enable key branch in dual-stream mode')
        parser.add_argument('--use_ctx_branch', type=int, default=1, choices=[0, 1],
                            help='whether to enable context branch in dual-stream mode')

        # ===== 新增：融合方式 =====
        parser.add_argument('--fusion_mode', type=str, default='avg',
                            choices=['avg', 'key', 'ctx', 'gate'],
                            help='fusion mode for dual-stream outputs')

        # 兼容保留：hard top-k 版本会用；soft-selection 版本暂不使用该参数
        parser.add_argument('--topk', default=4, type=int,
                            help='top-k instances for prototype anchor (hard top-k mode). '
                                 'In soft-selection mode, this is not used.')

        # soft selection 温度
        parser.add_argument('--score_tau', default=0.2, type=float,
                            help='temperature for soft instance selection in Stream-A '
                                 '(smaller => sparser weights; typical: 0.1~0.5)')

        # ----------------------
        # PGFE (Prototype Distance-Guided Feature Enhancement) - Stream-B only
        # ----------------------
        parser.add_argument('--use_pgfe', action='store_true',
                            help='enable PGFE (distance-guided enhancement) for Stream-B only')
        parser.add_argument('--no_pgfe', action='store_false', dest='use_pgfe',
                            help='disable PGFE')
        parser.set_defaults(use_pgfe=False)

        parser.add_argument('--pgfe_type', type=str, default='soft', choices=['soft', 'inv'],
                            help='PGFE weighting: soft (recommended) or inv (1/d)')

        parser.add_argument('--pgfe_gamma', type=float, default=0.2,
                            help='smoothing factor for PGFE soft mode (larger -> smoother). typical: 0.1~0.5')

        parser.add_argument('--proto_detach', action='store_true',
                            help='detach prototype when guiding Stream-B (recommended for stability)')
        parser.add_argument('--no_proto_detach', action='store_false', dest='proto_detach',
                            help='do not detach prototype for Stream-B')
        parser.set_defaults(proto_detach=True)

        # ----------------------
        # training hyperparameters
        # ----------------------
        parser.add_argument('--label_smoothing', default=0.0, type=float,
                            help='label smoothing (regression usually 0)')

        # 训练每个样本抽 K 个窗口
        parser.add_argument('--train_num_segs', default=1, type=int,
                            help='number of random segments/windows sampled per video in training (K). '
                                 'Set to 2/4 to increase training signal.')

        # sample_interval 可配（默认 3）
        parser.add_argument('--sample_interval', default=3, type=int,
                            help='frame sampling interval (e.g., 1/2/3). Smaller keeps more micro-dynamics.')

        # augmentation
        parser.add_argument('--random_sample', action='store_true',
                            help='enable random sampling/augmentation (if wired in dataloader)')
        parser.add_argument('--no_random_sample', action='store_false', dest='random_sample',
                            help='disable random sampling/augmentation')
        parser.set_defaults(random_sample=False)

        parser.add_argument('--color_jitter', default=0.4, type=float)

        # optimizer
        parser.add_argument('-o', '--optimizer', default="AdamW", type=str, metavar='Opti')

        # 回归 + 预训练微调：更稳的起步 lr / wd
        parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
        parser.add_argument('--wd', '--weight_decay', default=1e-2, type=float, metavar='W', dest='weight_decay')
        parser.add_argument('--eps', default=1e-8, type=float, metavar='EPSILON',
                            help='Optimizer Epsilon (default: 1e-8)')

        # 梯度裁剪
        parser.add_argument('--grad_clip', default=1.0, type=float,
                            help='max grad norm; set 0 to disable')

        # scheduler
        parser.add_argument('--lr_scheduler', default="cosine", type=str)
        parser.add_argument('--warmup_epochs', default=5, type=int)
        parser.add_argument('--min_lr', default=5e-6, type=float)
        parser.add_argument('--warmup_lr', default=0, type=float)

        return parser

    def parse(self):
        parser = self.initialize()
        args = parser.parse_args()

        # ----------------------
        # parse gpu ids
        # ----------------------
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            cur_id = int(str_id)
            if cur_id >= 0:
                args.gpu_ids.append(cur_id)

        # ----------------------
        # bool normalize
        # ----------------------
        args.use_key_branch = bool(args.use_key_branch)
        args.use_ctx_branch = bool(args.use_ctx_branch)

        # ----------------------
        # 自动生成实验名
        # ----------------------
        def get_variant_name(a):
            if not a.use_dual_stream:
                return "baseline"
            if a.use_key_branch and (not a.use_ctx_branch):
                return "key_only"
            if (not a.use_key_branch) and a.use_ctx_branch:
                return "ctx_only"
            if a.use_key_branch and a.use_ctx_branch:
                return f"full_{a.fusion_mode}"
            return "unknown"

        variant_name = get_variant_name(args)

        time_str = time.strftime("%m-%d-%H-%M")
        args.name = f"{args.dataset}_{args.task}_{variant_name}_{time_str}"

        args.output_path = os.path.join(args.output_root, args.name)
        os.makedirs(args.output_path, exist_ok=True)

        # ----------------------
        # dataset csv paths
        # ----------------------
        args.dev_dataset = None

        if args.dataset == "AVEC2014":
            args.train_dataset = os.path.join(args.data_root, "train_label.csv")
            args.dev_dataset = os.path.join(args.data_root, "dev_label.csv")
            args.test_dataset = os.path.join(args.data_root, "test_label.csv")

            for p in [args.train_dataset, args.dev_dataset, args.test_dataset]:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"CSV not found: {p}")

        elif args.dataset == "DFEW":
            args.train_dataset = os.path.join(
                args.data_root, "EmoLabel_DataSplit/train(single-labeled)/set_X.csv")
            args.test_dataset = os.path.join(
                args.data_root, "EmoLabel_DataSplit/test(single-labeled)/set_X.csv")
            args.train_dataset = args.train_dataset.replace('X', str(args.fold))
            args.test_dataset = args.test_dataset.replace('X', str(args.fold))

        elif args.dataset == "FERV39K":
            args.train_dataset = os.path.join(
                args.data_root, "FERV39K/FERV39k/4_setups/All_scenes/train_All.csv")
            args.test_dataset = os.path.join(
                args.data_root, "FERV39K/FERV39k/4_setups/All_scenes/test_All.csv")

        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        # ----------------------
        # safety checks
        # ----------------------
        if args.train_num_segs < 1:
            raise ValueError(f"--train_num_segs must be >= 1, got {args.train_num_segs}")
        if args.sample_interval < 1:
            raise ValueError(f"--sample_interval must be >= 1, got {args.sample_interval}")

        if args.topk < 1:
            raise ValueError(f"--topk must be >= 1, got {args.topk}")

        if args.score_tau <= 0:
            raise ValueError(f"--score_tau must be > 0, got {args.score_tau}")

        if args.instance_length < 1:
            raise ValueError(f"--instance_length must be >= 1, got {args.instance_length}")
        if args.num_frames % args.instance_length != 0:
            raise ValueError(
                f"--num_frames ({args.num_frames}) must be divisible by --instance_length ({args.instance_length}) "
                f"because bag_size = num_frames//instance_length is used by pwconv."
            )

        bag_size = args.num_frames // args.instance_length
        if args.topk > bag_size:
            print(f"[Warn] --topk ({args.topk}) > bag_size ({bag_size}). Will clamp topk to {bag_size} in model.")
        args.bag_size = bag_size

        # ----------------------
        # dual-stream / ablation checks
        # ----------------------
        if args.use_dual_stream:
            if (not args.use_key_branch) and (not args.use_ctx_branch):
                raise ValueError(
                    "When --use_dual_stream is enabled, at least one of "
                    "--use_key_branch / --use_ctx_branch must be 1."
                )

        if not args.use_dual_stream:
            if (not args.use_key_branch) or (not args.use_ctx_branch):
                print("[Warn] Dual-stream is disabled, so --use_key_branch / --use_ctx_branch will be ignored.")

        if args.fusion_mode in ["key", "ctx"] and not args.use_dual_stream:
            print(f"[Warn] fusion_mode={args.fusion_mode} has no effect when dual-stream is disabled.")

        if args.fusion_mode == "gate" and not (args.use_key_branch and args.use_ctx_branch):
            print("[Warn] fusion_mode=gate is designed for full dual-branch setting. "
                  "In single-branch ablation it will be ignored.")

        # ----------------------
        # PGFE checks
        # ----------------------
        if args.use_pgfe:
            if args.pgfe_gamma <= 0:
                raise ValueError(f"--pgfe_gamma must be > 0, got {args.pgfe_gamma}")
            if not args.use_dual_stream:
                print("[Warn] --use_pgfe is enabled but --use_dual_stream is False. "
                      "PGFE will have no effect (because prototype is not used).")
            if not args.use_ctx_branch:
                print("[Warn] --use_pgfe is enabled but context branch is disabled. "
                      "PGFE will have no effect because it is designed for Stream-B.")

        # ----------------------
        # info hints
        # ----------------------
        print(f"[INFO] Experiment variant: {variant_name}")

        if args.use_dual_stream:
            print(f"[INFO] Dual-stream enabled. score_tau={args.score_tau}, "
                  f"use_key_branch={args.use_key_branch}, "
                  f"use_ctx_branch={args.use_ctx_branch}, "
                  f"fusion_mode={args.fusion_mode}")
            print(f"[INFO] Note: --topk={args.topk} is not used in soft-selection mode.")

        if args.use_dual_stream and args.use_pgfe:
            print(f"[INFO] PGFE enabled for Stream-B. type={args.pgfe_type}, gamma={args.pgfe_gamma}, "
                  f"proto_detach={args.proto_detach}")

        return args