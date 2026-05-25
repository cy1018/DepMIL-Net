 # 2026年2月23日修改代码 + Dual-stream(top-k prototype anchor) 版本（AVEC2014 回归）
# 2026-02-23 基础版 + 修复版 Dual-stream（可微 soft selection）
# 关键改动：
# 1) MIL() 返回 (inst_feat, attn_score)，其中 attn_score 来自 MHSA attention 的聚合
# 2) Dual-stream 的 Stream-A 不再用 hard topk+gather（不可微），改为 soft weights 构造 prototype（可微）
# 3) Stream-B 用 prototype 引导相似度加权聚合 context
# 4) 保留 baseline（pwconv+fc）分支，便于消融
#最终版的代码
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchvision.models.video import r3d_18, R3D_18_Weights
# from einops import rearrange

# from utils import *  # 需要你工程里已有的 DMIN 等


# class M3DFEL(nn.Module):
#     def __init__(self, args):
#         super(M3DFEL, self).__init__()

#         self.args = args
#         self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')

#         # bag_size = num_frames // instance_length
#         self.bag_size = self.args.num_frames // self.args.instance_length
#         self.instance_length = self.args.instance_length

#         # backbone networks
#         model = r3d_18(weights=R3D_18_Weights.DEFAULT)
#         self.features = nn.Sequential(*list(model.children())[:-1])  # output: [B,512,1,1,1]

#         # BiLSTM (instance contextualizer)
#         self.lstm = nn.LSTM(
#             input_size=512, hidden_size=512,
#             num_layers=2, batch_first=True, bidirectional=True
#         )

#         # multi-head self attention (MHSA)
#         self.heads = 8
#         self.dim_head = 1024 // self.heads
#         self.scale = self.dim_head ** -0.5
#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(1024, (self.dim_head * self.heads) * 3, bias=False)

#         # stabilization
#         self.norm = DMIN(num_features=1024)

#         # baseline pooling (depends on fixed bag_size=N)
#         self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
#         self.fc = nn.Linear(1024, self.args.num_classes)  # regression: num_classes=1

#         # ===== Dual-stream settings =====
#         self.use_dual_stream = getattr(self.args, "use_dual_stream", False)

#         # soft selection temperature (smaller => sparser, larger => smoother)
#         self.score_tau = float(getattr(self.args, "score_tau", 0.2))

#         # Stream-A / Stream-B regressors
#         self.reg_key = nn.Linear(1024, self.args.num_classes)
#         self.reg_ctx = nn.Linear(1024, self.args.num_classes)

#         # (optional) gate fusion if you want later; default not used
#         self.fuse_gate = nn.Sequential(
#             nn.Linear(2048, 1),
#             nn.Sigmoid()
#         )

#     def MIL(self, x):
#         """
#         Inputs:
#             x: [B, N, 512]
#         Returns:
#             inst_feat: [B, N, 1024]  (contextual instance features)
#             attn_score: [B, N]       (instance importance score derived from MHSA attention)
#         """
#         self.lstm.flatten_parameters()
#         x, _ = self.lstm(x)  # [B, N, 1024]
#         ori_x = x

#         # MHSA
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, H, N, N]
#         attn = self.attend(dots)                                   # [B, H, N, N]
#         x = torch.matmul(attn, v)                                  # [B, H, N, Dh]
#         x = rearrange(x, 'b h n d -> b n (h d)')                    # [B, N, 1024]

#         # stabilization gate
#         x = self.norm(x)
#         x = torch.sigmoid(x)
#         inst_feat = ori_x * x                                       # [B, N, 1024]

#         # derive instance score from attention:
#         # mean over heads, mean over query dimension -> each key's average attention mass
#         # attn: [B,H,N,N] -> [B,N]
#         attn_score = attn.mean(dim=1).mean(dim=1)

#         return inst_feat, attn_score

#     def forward(self, x):
#         """
#         支持两种输入格式：
#         1) x: [B, num_frames, C, H, W]    (num_frames = bag_size * instance_length)
#         2) x: [B, bag_size, C, il, H, W] (MIL dataloader 输出)

#         最终统一成 [(B*bag_size), C, il, H, W] 送入3D backbone
#         """
#         if x.dim() == 5:
#             # x: [B, T, C, H, W]
#             B, T, C, H, W = x.shape
#             il = self.instance_length

#             if T % il != 0:
#                 # fallback infer il using self.bag_size
#                 if self.bag_size <= 0 or T % self.bag_size != 0:
#                     raise ValueError(
#                         f"Cannot infer instance_length/bag_size from input: T={T}, "
#                         f"instance_length={self.instance_length}, bag_size={self.bag_size}"
#                     )
#                 bag_size = self.bag_size
#                 il = T // bag_size
#             else:
#                 bag_size = T // il

#             x = rearrange(
#                 x,
#                 'b (t1 t2) c h w -> (b t1) c t2 h w',
#                 t1=bag_size,
#                 t2=il
#             )
#             bag_size_infer = bag_size

#         elif x.dim() == 6:
#             # x: [B, N, C, t, H, W]
#             B, N, C, t, H, W = x.shape
#             if N != self.bag_size:
#                 raise ValueError(
#                     f"Input bag_size N={N} != model.bag_size={self.bag_size}. "
#                     f"Check args.num_frames//args.instance_length and dataloader bag_size."
#                 )
#             x = rearrange(
#                 x,
#                 'b t1 c t2 h w -> (b t1) c t2 h w',
#                 t1=N,
#                 t2=t
#             )
#             bag_size_infer = N

#         else:
#             raise ValueError(f"Unexpected input dims: {x.dim()}, expected 5 or 6. Got shape={tuple(x.shape)}")

#         # backbone: -> [B*bag, 512, 1, 1, 1] -> flatten -> [B*bag, 512]
#         x = self.features(x)
#         x = x.flatten(1)

#         # -> [B, N, 512]
#         x = rearrange(x, '(b t) c -> b t c', t=bag_size_infer)

#         # -> [B, N, 1024] and attention-based scores [B, N]
#         inst_feat, attn_score = self.MIL(x)

#         # ===== baseline path (single-stream pooling) =====
#         if not self.use_dual_stream:
#             bag_feat = self.pwconv(inst_feat).squeeze(1)  # [B, 1024]
#             out = self.fc(bag_feat)                       # [B, 1]
#             if out.dim() == 1:
#                 out = out.unsqueeze(1)
#             return out

#         # ===== dual-stream path (ours) =====
#         # Stream-A: soft selection prototype (differentiable)
#         # w_key: [B,N]
#         w_key = torch.softmax(attn_score / self.score_tau, dim=1)
#         proto = (inst_feat * w_key.unsqueeze(-1)).sum(dim=1)         # [B, 1024]
#         y_key = self.reg_key(proto)                                  # [B, 1]

#         # Stream-B: prototype-guided context pooling
#         proto_norm = F.normalize(proto, dim=-1)                      # [B,1024]
#         feat_norm = F.normalize(inst_feat, dim=-1)                   # [B,N,1024]
#         sim = (feat_norm * proto_norm.unsqueeze(1)).sum(dim=-1)      # [B,N]
#         w_ctx = torch.softmax(sim, dim=1)                            # [B,N]
#         ctx = (inst_feat * w_ctx.unsqueeze(-1)).sum(dim=1)           # [B,1024]
#         y_ctx = self.reg_ctx(ctx)                                    # [B,1]

#         # Fusion: stable parameter-free average (recommended first)
#         out = 0.5 * y_key + 0.5 * y_ctx

#         # If you later want gate fusion, replace the above with:
#         # gate = self.fuse_gate(torch.cat([proto, ctx], dim=-1))      # [B,1]
#         # out = gate * y_key + (1 - gate) * y_ctx

#         if out.dim() == 1:
#             out = out.unsqueeze(1)

#         return out
#最终版的代码完


# #  为做消融实验修改的代码
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchvision.models.video import r3d_18, R3D_18_Weights
# from einops import rearrange

# from utils import *  # 需要你工程里已有的 DMIN 等


# class M3DFEL(nn.Module):
#     def __init__(self, args):
#         super(M3DFEL, self).__init__()

#         self.args = args
#         self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')

#         # bag_size = num_frames // instance_length
#         self.bag_size = self.args.num_frames // self.args.instance_length
#         self.instance_length = self.args.instance_length

#         # backbone networks
#         model = r3d_18(weights=R3D_18_Weights.DEFAULT)
#         self.features = nn.Sequential(*list(model.children())[:-1])  # output: [B,512,1,1,1]

#         # BiLSTM (instance contextualizer)
#         self.lstm = nn.LSTM(
#             input_size=512,
#             hidden_size=512,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )

#         # multi-head self attention (MHSA)
#         self.heads = 8
#         self.dim_head = 1024 // self.heads
#         self.scale = self.dim_head ** -0.5
#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(1024, (self.dim_head * self.heads) * 3, bias=False)

#         # stabilization
#         self.norm = DMIN(num_features=1024)

#         # baseline pooling (depends on fixed bag_size=N)
#         self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
#         self.fc = nn.Linear(1024, self.args.num_classes)  # regression: num_classes=1

#         # ===== Dual-stream settings =====
#         self.use_dual_stream = bool(getattr(self.args, "use_dual_stream", False))

#         # ablation switches
#         self.use_key_branch = bool(getattr(self.args, "use_key_branch", True))
#         self.use_ctx_branch = bool(getattr(self.args, "use_ctx_branch", True))

#         # fusion mode: "avg" / "key" / "ctx" / "gate"
#         self.fusion_mode = getattr(self.args, "fusion_mode", "avg")

#         # soft selection temperature (smaller => sparser, larger => smoother)
#         self.score_tau = float(getattr(self.args, "score_tau", 0.2))

#         # Stream-A / Stream-B regressors
#         self.reg_key = nn.Linear(1024, self.args.num_classes)
#         self.reg_ctx = nn.Linear(1024, self.args.num_classes)

#         # optional gate fusion
#         self.fuse_gate = nn.Sequential(
#             nn.Linear(2048, 1),
#             nn.Sigmoid()
#         )

#     def MIL(self, x):
#         """
#         Inputs:
#             x: [B, N, 512]
#         Returns:
#             inst_feat: [B, N, 1024]  (contextual instance features)
#             attn_score: [B, N]       (instance importance score derived from MHSA attention)
#         """
#         self.lstm.flatten_parameters()
#         x, _ = self.lstm(x)  # [B, N, 1024]
#         ori_x = x

#         # MHSA
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, H, N, N]
#         attn = self.attend(dots)                                   # [B, H, N, N]
#         x = torch.matmul(attn, v)                                  # [B, H, N, Dh]
#         x = rearrange(x, 'b h n d -> b n (h d)')                   # [B, N, 1024]

#         # stabilization gate
#         x = self.norm(x)
#         x = torch.sigmoid(x)
#         inst_feat = ori_x * x                                      # [B, N, 1024]

#         # derive instance score from attention:
#         # mean over heads, mean over query dimension -> each key's average attention mass
#         # attn: [B, H, N, N] -> [B, N]
#         attn_score = attn.mean(dim=1).mean(dim=1)

#         return inst_feat, attn_score

#     def forward(self, x):
#         """
#         支持两种输入格式：
#         1) x: [B, num_frames, C, H, W]    (num_frames = bag_size * instance_length)
#         2) x: [B, bag_size, C, il, H, W]  (MIL dataloader 输出)

#         最终统一成 [(B*bag_size), C, il, H, W] 送入3D backbone
#         """
#         if x.dim() == 5:
#             # x: [B, T, C, H, W]
#             B, T, C, H, W = x.shape
#             il = self.instance_length

#             if T % il != 0:
#                 # fallback infer il using self.bag_size
#                 if self.bag_size <= 0 or T % self.bag_size != 0:
#                     raise ValueError(
#                         f"Cannot infer instance_length/bag_size from input: T={T}, "
#                         f"instance_length={self.instance_length}, bag_size={self.bag_size}"
#                     )
#                 bag_size = self.bag_size
#                 il = T // bag_size
#             else:
#                 bag_size = T // il

#             x = rearrange(
#                 x,
#                 'b (t1 t2) c h w -> (b t1) c t2 h w',
#                 t1=bag_size,
#                 t2=il
#             )
#             bag_size_infer = bag_size

#         elif x.dim() == 6:
#             # x: [B, N, C, t, H, W]
#             B, N, C, t, H, W = x.shape
#             if N != self.bag_size:
#                 raise ValueError(
#                     f"Input bag_size N={N} != model.bag_size={self.bag_size}. "
#                     f"Check args.num_frames//args.instance_length and dataloader bag_size."
#                 )
#             x = rearrange(
#                 x,
#                 'b t1 c t2 h w -> (b t1) c t2 h w',
#                 t1=N,
#                 t2=t
#             )
#             bag_size_infer = N

#         else:
#             raise ValueError(
#                 f"Unexpected input dims: {x.dim()}, expected 5 or 6. Got shape={tuple(x.shape)}"
#             )

#         # backbone: -> [B*bag, 512, 1, 1, 1] -> flatten -> [B*bag, 512]
#         x = self.features(x)
#         x = x.flatten(1)

#         # -> [B, N, 512]
#         x = rearrange(x, '(b t) c -> b t c', t=bag_size_infer)

#         # -> [B, N, 1024] and attention-based scores [B, N]
#         inst_feat, attn_score = self.MIL(x)

#         # ===== baseline path (single-stream pooling) =====
#         if not self.use_dual_stream:
#             bag_feat = self.pwconv(inst_feat).squeeze(1)  # [B, 1024]
#             out = self.fc(bag_feat)                       # [B, 1]
#             if out.dim() == 1:
#                 out = out.unsqueeze(1)
#             return out

#         # dual-stream 模式下，至少要保留一个分支
#         if (not self.use_key_branch) and (not self.use_ctx_branch):
#             raise ValueError("In dual-stream mode, at least one branch must be enabled.")

#         proto = None
#         ctx = None
#         y_key = None
#         y_ctx = None

#         # ===== Stream-A: Key branch =====
#         if self.use_key_branch:
#             # soft selection prototype (differentiable)
#             w_key = torch.softmax(attn_score / self.score_tau, dim=1)   # [B, N]
#             proto = (inst_feat * w_key.unsqueeze(-1)).sum(dim=1)        # [B, 1024]
#             y_key = self.reg_key(proto)                                 # [B, 1]

#         # ===== Stream-B: Context branch =====
#         if self.use_ctx_branch:
#             # 如果 key branch 被关掉，则用 mean instance feature 作为 surrogate prototype
#             if proto is None:
#                 proto = inst_feat.mean(dim=1)                           # [B, 1024]

#             proto_norm = F.normalize(proto, dim=-1)                     # [B, 1024]
#             feat_norm = F.normalize(inst_feat, dim=-1)                  # [B, N, 1024]
#             sim = (feat_norm * proto_norm.unsqueeze(1)).sum(dim=-1)     # [B, N]
#             w_ctx = torch.softmax(sim, dim=1)                           # [B, N]
#             ctx = (inst_feat * w_ctx.unsqueeze(-1)).sum(dim=1)          # [B, 1024]
#             y_ctx = self.reg_ctx(ctx)                                   # [B, 1]

#         # ===== Output selection / fusion =====
#         if self.use_key_branch and not self.use_ctx_branch:
#             # Key branch only
#             out = y_key

#         elif (not self.use_key_branch) and self.use_ctx_branch:
#             # Context branch only
#             out = y_ctx

#         else:
#             # Full ProtoDual
#             if self.fusion_mode == "avg":
#                 out = 0.5 * y_key + 0.5 * y_ctx

#             elif self.fusion_mode == "key":
#                 out = y_key

#             elif self.fusion_mode == "ctx":
#                 out = y_ctx

#             elif self.fusion_mode == "gate":
#                 if ctx is None:
#                     proto_norm = F.normalize(proto, dim=-1)
#                     feat_norm = F.normalize(inst_feat, dim=-1)
#                     sim = (feat_norm * proto_norm.unsqueeze(1)).sum(dim=-1)
#                     w_ctx = torch.softmax(sim, dim=1)
#                     ctx = (inst_feat * w_ctx.unsqueeze(-1)).sum(dim=1)

#                 gate = self.fuse_gate(torch.cat([proto, ctx], dim=-1))  # [B, 1]
#                 out = gate * y_key + (1 - gate) * y_ctx

#             else:
#                 raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

#         if out.dim() == 1:
#             out = out.unsqueeze(1)

#         return out



# #为了做可视化，改的内容
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchvision.models.video import r3d_18, R3D_18_Weights
# from einops import rearrange

# from utils import *   # 这里保留你工程里的 DMIN 等


# class M3DFEL(nn.Module):
#     def __init__(self, args):
#         super(M3DFEL, self).__init__()

#         self.args = args
#         self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')

#         # bag_size = num_frames // instance_length
#         self.bag_size = self.args.num_frames // self.args.instance_length
#         self.instance_length = self.args.instance_length

#         # backbone
#         model = r3d_18(weights=R3D_18_Weights.DEFAULT)
#         self.features = nn.Sequential(*list(model.children())[:-1])  # [B,512,1,1,1]

#         # BiLSTM
#         self.lstm = nn.LSTM(
#             input_size=512,
#             hidden_size=512,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )

#         # MHSA
#         self.heads = 8
#         self.dim_head = 1024 // self.heads
#         self.scale = self.dim_head ** -0.5
#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(1024, (self.dim_head * self.heads) * 3, bias=False)

#         # stabilization
#         self.norm = DMIN(num_features=1024)

#         # baseline pooling
#         self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
#         self.fc = nn.Linear(1024, self.args.num_classes)

#         # dual-stream settings
#         self.use_dual_stream = getattr(self.args, "use_dual_stream", False)
#         self.score_tau = float(getattr(self.args, "score_tau", 0.2))

#         self.reg_key = nn.Linear(1024, self.args.num_classes)
#         self.reg_ctx = nn.Linear(1024, self.args.num_classes)

#         self.fuse_gate = nn.Sequential(
#             nn.Linear(2048, 1),
#             nn.Sigmoid()
#         )

#     def MIL(self, x):
#         """
#         Inputs:
#             x: [B, N, 512]
#         Returns:
#             inst_feat: [B, N, 1024]
#             attn_score: [B, N]
#         """
#         self.lstm.flatten_parameters()
#         x, _ = self.lstm(x)     # [B, N, 1024]
#         ori_x = x

#         # MHSA
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B,H,N,N]
#         attn = self.attend(dots)                                   # [B,H,N,N]
#         x = torch.matmul(attn, v)                                  # [B,H,N,Dh]
#         x = rearrange(x, 'b h n d -> b n (h d)')                   # [B,N,1024]

#         # stabilization
#         x = self.norm(x)
#         x = torch.sigmoid(x)
#         inst_feat = ori_x * x                                      # [B,N,1024]

#         # attention-derived instance score
#         attn_score = attn.mean(dim=1).mean(dim=1)                  # [B,N]

#         return inst_feat, attn_score

#     def forward(self, x, return_interpret=False):
#         """
#         支持两种输入格式：
#         1) x: [B, T, C, H, W]
#         2) x: [B, N, C, t, H, W]

#         return_interpret:
#             False -> 只返回 pred
#             True  -> 返回 (pred, aux)
#         """
#         if x.dim() == 5:
#             # [B, T, C, H, W]
#             B, T, C, H, W = x.shape
#             il = self.instance_length

#             if T % il != 0:
#                 if self.bag_size <= 0 or T % self.bag_size != 0:
#                     raise ValueError(
#                         f"Cannot infer instance_length/bag_size from input: "
#                         f"T={T}, instance_length={self.instance_length}, bag_size={self.bag_size}"
#                     )
#                 bag_size = self.bag_size
#                 il = T // bag_size
#             else:
#                 bag_size = T // il

#             x = rearrange(
#                 x,
#                 'b (t1 t2) c h w -> (b t1) c t2 h w',
#                 t1=bag_size,
#                 t2=il
#             )
#             bag_size_infer = bag_size

#         elif x.dim() == 6:
#             # [B, N, C, t, H, W]
#             B, N, C, t, H, W = x.shape
#             if N != self.bag_size:
#                 raise ValueError(
#                     f"Input bag_size N={N} != model.bag_size={self.bag_size}. "
#                     f"Check args.num_frames//args.instance_length and dataloader bag_size."
#                 )
#             x = rearrange(
#                 x,
#                 'b t1 c t2 h w -> (b t1) c t2 h w',
#                 t1=N,
#                 t2=t
#             )
#             bag_size_infer = N

#         else:
#             raise ValueError(f"Unexpected input dims: {x.dim()}, got shape={tuple(x.shape)}")

#         # backbone
#         x = self.features(x)
#         x = x.flatten(1)                                           # [(B*N),512]

#         # [B,N,512]
#         x = rearrange(x, '(b t) c -> b t c', t=bag_size_infer)

#         # MIL
#         inst_feat, attn_score = self.MIL(x)                        # [B,N,1024], [B,N]

#         # baseline instance-level weights
#         baseline_weights = torch.softmax(attn_score, dim=1)        # [B,N]

#         # ===== baseline path =====
#         if not self.use_dual_stream:
#             bag_feat = self.pwconv(inst_feat).squeeze(1)           # [B,1024]
#             out = self.fc(bag_feat)                                # [B,1]

#             if out.dim() == 1:
#                 out = out.unsqueeze(1)

#             if return_interpret:
#                 aux = {
#                     "attn_score": attn_score.detach(),
#                     "baseline_weights": baseline_weights.detach(),
#                     "inst_feat": inst_feat.detach(),
#                 }
#                 return out, aux

#             return out

#         # ===== dual-stream path =====
#         # Stream-A: key prototype
#         w_key = torch.softmax(attn_score / self.score_tau, dim=1)  # [B,N]
#         proto = (inst_feat * w_key.unsqueeze(-1)).sum(dim=1)       # [B,1024]
#         y_key = self.reg_key(proto)                                # [B,1]

#         # Stream-B: prototype-guided context
#         proto_norm = F.normalize(proto, dim=-1)                    # [B,1024]
#         feat_norm = F.normalize(inst_feat, dim=-1)                 # [B,N,1024]
#         sim = (feat_norm * proto_norm.unsqueeze(1)).sum(dim=-1)    # [B,N]
#         w_ctx = torch.softmax(sim, dim=1)                          # [B,N]
#         ctx = (inst_feat * w_ctx.unsqueeze(-1)).sum(dim=1)         # [B,1024]
#         y_ctx = self.reg_ctx(ctx)                                  # [B,1]

#         # fusion
#         out = 0.5 * y_key + 0.5 * y_ctx

#         if out.dim() == 1:
#             out = out.unsqueeze(1)

#         if return_interpret:
#             aux = {
#                 "attn_score": attn_score.detach(),           # 原始实例分数
#                 "baseline_weights": baseline_weights.detach(),  # baseline线
#                 "key_weights": w_key.detach(),               # key branch线
#                 "context_scores": sim.detach(),              # context原始相似度
#                 "context_weights": w_ctx.detach(),           # context branch线
#                 "proto": proto.detach(),
#                 "ctx": ctx.detach(),
#                 "inst_feat": inst_feat.detach(),
#             }
#             return out, aux

#         return out


#为了做时序可视化图
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchvision.models.video import r3d_18, R3D_18_Weights
# from einops import rearrange

# from utils import *   # 保留你工程里的 DMIN 等


# class M3DFEL(nn.Module):
#     def __init__(self, args):
#         super(M3DFEL, self).__init__()

#         self.args = args
#         self.device = torch.device(
#             'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu'
#         )

#         # bag_size = num_frames // instance_length
#         self.bag_size = self.args.num_frames // self.args.instance_length
#         self.instance_length = self.args.instance_length

#         # backbone
#         model = r3d_18(weights=R3D_18_Weights.DEFAULT)
#         self.features = nn.Sequential(*list(model.children())[:-1])  # [B,512,1,1,1]

#         # BiLSTM
#         self.lstm = nn.LSTM(
#             input_size=512,
#             hidden_size=512,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )

#         # MHSA
#         self.heads = 8
#         self.dim_head = 1024 // self.heads
#         self.scale = self.dim_head ** -0.5
#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(1024, (self.dim_head * self.heads) * 3, bias=False)

#         # stabilization / ASM-like gate
#         self.norm = DMIN(num_features=1024)

#         # baseline pooling
#         self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
#         self.fc = nn.Linear(1024, self.args.num_classes)

#         # dual-stream settings
#         self.use_dual_stream = getattr(self.args, "use_dual_stream", False)
#         self.score_tau = float(getattr(self.args, "score_tau", 0.2))

#         self.reg_key = nn.Linear(1024, self.args.num_classes)
#         self.reg_ctx = nn.Linear(1024, self.args.num_classes)

#         self.fuse_gate = nn.Sequential(
#             nn.Linear(2048, 1),
#             nn.Sigmoid()
#         )

#     def MIL(self, x, return_aux=False):
#         """
#         Inputs:
#             x: [B, N, 512]

#         Returns:
#             inst_feat:  [B, N, 1024]
#             attn_score: [B, N]

#         If return_aux=True, also returns:
#             bilstm_feat: [B, N, 1024]
#             mhsa_feat:   [B, N, 1024]
#             asm_gate:    [B, N, 1024]
#             attn_map:    [B, N, N]
#         """
#         self.lstm.flatten_parameters()

#         # 1) BiLSTM output
#         bilstm_feat, _ = self.lstm(x)  # [B, N, 1024]

#         # 2) MHSA
#         qkv = self.to_qkv(bilstm_feat).chunk(3, dim=-1)
#         q, k, v = map(
#             lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
#             qkv
#         )

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, H, N, N]
#         attn = self.attend(dots)                                   # [B, H, N, N]

#         mhsa_feat = torch.matmul(attn, v)                          # [B, H, N, Dh]
#         mhsa_feat = rearrange(mhsa_feat, 'b h n d -> b n (h d)')   # [B, N, 1024]

#         # 3) ASM-like gate
#         asm_gate = self.norm(mhsa_feat)
#         asm_gate = torch.sigmoid(asm_gate)                         # [B, N, 1024]

#         # 4) enhanced instance feature
#         inst_feat = bilstm_feat * asm_gate                         # [B, N, 1024]

#         # 5) attention-derived instance score
#         attn_map = attn.mean(dim=1)                                # [B, N, N]
#         attn_score = attn_map.mean(dim=1)                          # [B, N]

#         if return_aux:
#             aux = {
#                 "bilstm_feat": bilstm_feat.detach(),
#                 "mhsa_feat": mhsa_feat.detach(),
#                 "asm_gate": asm_gate.detach(),
#                 "attn_map": attn_map.detach(),
#             }
#             return inst_feat, attn_score, aux

#         return inst_feat, attn_score

#     def forward(self, x, return_interpret=False):
#         """
#         支持两种输入格式：
#         1) x: [B, T, C, H, W]
#         2) x: [B, N, C, t, H, W]

#         return_interpret:
#             False -> 只返回 pred
#             True  -> 返回 (pred, aux)
#         """
#         if x.dim() == 5:
#             # [B, T, C, H, W]
#             B, T, C, H, W = x.shape
#             il = self.instance_length

#             if T % il != 0:
#                 if self.bag_size <= 0 or T % self.bag_size != 0:
#                     raise ValueError(
#                         f"Cannot infer instance_length/bag_size from input: "
#                         f"T={T}, instance_length={self.instance_length}, bag_size={self.bag_size}"
#                     )
#                 bag_size = self.bag_size
#                 il = T // bag_size
#             else:
#                 bag_size = T // il

#             x = rearrange(
#                 x,
#                 'b (t1 t2) c h w -> (b t1) c t2 h w',
#                 t1=bag_size,
#                 t2=il
#             )
#             bag_size_infer = bag_size

#         elif x.dim() == 6:
#             # [B, N, C, t, H, W]
#             B, N, C, t, H, W = x.shape
#             if N != self.bag_size:
#                 raise ValueError(
#                     f"Input bag_size N={N} != model.bag_size={self.bag_size}. "
#                     f"Check args.num_frames//args.instance_length and dataloader bag_size."
#                 )
#             x = rearrange(
#                 x,
#                 'b t1 c t2 h w -> (b t1) c t2 h w',
#                 t1=N,
#                 t2=t
#             )
#             bag_size_infer = N

#         else:
#             raise ValueError(f"Unexpected input dims: {x.dim()}, got shape={tuple(x.shape)}")

#         # backbone
#         x = self.features(x)
#         x = x.flatten(1)                                           # [(B*N), 512]

#         # reshape to [B, N, 512]
#         x = rearrange(x, '(b t) c -> b t c', t=bag_size_infer)

#         # MIL
#         if return_interpret:
#             inst_feat, attn_score, mil_aux = self.MIL(x, return_aux=True)
#         else:
#             inst_feat, attn_score = self.MIL(x, return_aux=False)
#             mil_aux = None

#         # baseline instance-level weights
#         baseline_weights = torch.softmax(attn_score, dim=1)        # [B, N]

#         # ===== baseline path =====
#         if not self.use_dual_stream:
#             bag_feat = self.pwconv(inst_feat).squeeze(1)           # [B, 1024]
#             out = self.fc(bag_feat)                                # [B, num_classes]

#             if out.dim() == 1:
#                 out = out.unsqueeze(1)

#             if return_interpret:
#                 aux = {
#                     "attn_score": attn_score.detach(),
#                     "baseline_weights": baseline_weights.detach(),
#                     "inst_feat": inst_feat.detach(),

#                     "bilstm_feat": mil_aux["bilstm_feat"],
#                     "mhsa_feat": mil_aux["mhsa_feat"],
#                     "asm_gate": mil_aux["asm_gate"],
#                     "attn_map": mil_aux["attn_map"],
#                 }
#                 return out, aux

#             return out

#         # ===== dual-stream path =====
#         # Stream-A: key prototype
#         w_key = torch.softmax(attn_score / self.score_tau, dim=1)  # [B, N]
#         proto = (inst_feat * w_key.unsqueeze(-1)).sum(dim=1)       # [B, 1024]
#         y_key = self.reg_key(proto)                                # [B, num_classes]

#         # Stream-B: prototype-guided context
#         proto_norm = F.normalize(proto, dim=-1)                    # [B, 1024]
#         feat_norm = F.normalize(inst_feat, dim=-1)                 # [B, N, 1024]
#         sim = (feat_norm * proto_norm.unsqueeze(1)).sum(dim=-1)    # [B, N]
#         w_ctx = torch.softmax(sim, dim=1)                          # [B, N]
#         ctx = (inst_feat * w_ctx.unsqueeze(-1)).sum(dim=1)         # [B, 1024]
#         y_ctx = self.reg_ctx(ctx)                                  # [B, num_classes]

#         # fusion
#         out = 0.5 * y_key + 0.5 * y_ctx

#         if out.dim() == 1:
#             out = out.unsqueeze(1)

#         if return_interpret:
#             aux = {
#                 "attn_score": attn_score.detach(),             # 原始实例分数
#                 "baseline_weights": baseline_weights.detach(), # baseline实例权重

#                 "key_weights": w_key.detach(),                 # key branch权重
#                 "context_scores": sim.detach(),                # context相似度
#                 "context_weights": w_ctx.detach(),             # context branch权重

#                 "proto": proto.detach(),
#                 "ctx": ctx.detach(),
#                 "inst_feat": inst_feat.detach(),

#                 "bilstm_feat": mil_aux["bilstm_feat"],
#                 "mhsa_feat": mil_aux["mhsa_feat"],
#                 "asm_gate": mil_aux["asm_gate"],
#                 "attn_map": mil_aux["attn_map"],
#             }
#             return out, aux

#         return out





# 为删掉权重占比高的实例改的代码
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchvision.models.video import r3d_18, R3D_18_Weights
# from einops import rearrange

# from utils import *   # 保留你工程里的 DMIN 等


# class M3DFEL(nn.Module):
#     def __init__(self, args):
#         super(M3DFEL, self).__init__()

#         self.args = args
#         self.device = torch.device(
#             'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu'
#         )

#         # bag_size = num_frames // instance_length
#         self.bag_size = self.args.num_frames // self.args.instance_length
#         self.instance_length = self.args.instance_length

#         # backbone
#         model = r3d_18(weights=R3D_18_Weights.DEFAULT)
#         self.features = nn.Sequential(*list(model.children())[:-1])  # [B,512,1,1,1]

#         # BiLSTM
#         self.lstm = nn.LSTM(
#             input_size=512,
#             hidden_size=512,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )

#         # MHSA
#         self.heads = 8
#         self.dim_head = 1024 // self.heads
#         self.scale = self.dim_head ** -0.5
#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(1024, (self.dim_head * self.heads) * 3, bias=False)

#         # stabilization / ASM-like gate
#         self.norm = DMIN(num_features=1024)

#         # baseline pooling
#         self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
#         self.fc = nn.Linear(1024, self.args.num_classes)

#         # dual-stream settings
#         self.use_dual_stream = getattr(self.args, "use_dual_stream", False)
#         self.score_tau = float(getattr(self.args, "score_tau", 0.2))

#         self.reg_key = nn.Linear(1024, self.args.num_classes)
#         self.reg_ctx = nn.Linear(1024, self.args.num_classes)

#         self.fuse_gate = nn.Sequential(
#             nn.Linear(2048, 1),
#             nn.Sigmoid()
#         )

#     @staticmethod
#     def masked_softmax(logits, keep_mask=None, dim=1):
#         """
#         logits:    [B, N]
#         keep_mask: [B, N], True=保留, False=移除
#         """
#         if keep_mask is None:
#             return torch.softmax(logits, dim=dim)

#         if keep_mask.dtype != torch.bool:
#             keep_mask = keep_mask.bool()

#         masked_logits = logits.masked_fill(~keep_mask, -1e9)
#         return torch.softmax(masked_logits, dim=dim)

#     def MIL(self, x, return_aux=False):
#         """
#         Inputs:
#             x: [B, N, 512]

#         Returns:
#             inst_feat:  [B, N, 1024]
#             attn_score: [B, N]

#         If return_aux=True, also returns:
#             bilstm_feat: [B, N, 1024]
#             mhsa_feat:   [B, N, 1024]
#             asm_gate:    [B, N, 1024]
#             attn_map:    [B, N, N]
#         """
#         self.lstm.flatten_parameters()

#         # 1) BiLSTM output
#         bilstm_feat, _ = self.lstm(x)  # [B, N, 1024]

#         # 2) MHSA
#         qkv = self.to_qkv(bilstm_feat).chunk(3, dim=-1)
#         q, k, v = map(
#             lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
#             qkv
#         )

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, H, N, N]
#         attn = self.attend(dots)                                   # [B, H, N, N]

#         mhsa_feat = torch.matmul(attn, v)                          # [B, H, N, Dh]
#         mhsa_feat = rearrange(mhsa_feat, 'b h n d -> b n (h d)')   # [B, N, 1024]

#         # 3) ASM-like gate
#         asm_gate = self.norm(mhsa_feat)
#         asm_gate = torch.sigmoid(asm_gate)                         # [B, N, 1024]

#         # 4) enhanced instance feature
#         inst_feat = bilstm_feat * asm_gate                         # [B, N, 1024]

#         # 5) attention-derived instance score
#         attn_map = attn.mean(dim=1)                                # [B, N, N]
#         attn_score = attn_map.mean(dim=1)                          # [B, N]

#         if return_aux:
#             aux = {
#                 "bilstm_feat": bilstm_feat.detach(),
#                 "mhsa_feat": mhsa_feat.detach(),
#                 "asm_gate": asm_gate.detach(),
#                 "attn_map": attn_map.detach(),
#             }
#             return inst_feat, attn_score, aux

#         return inst_feat, attn_score

#     def forward(self, x, return_interpret=False, instance_keep_mask=None):
#         """
#         支持两种输入格式：
#         1) x: [B, T, C, H, W]
#         2) x: [B, N, C, t, H, W]

#         instance_keep_mask:
#             None 或 [B, N]
#             True=保留, False=移除

#         return_interpret:
#             False -> 只返回 pred
#             True  -> 返回 (pred, aux)
#         """
#         if x.dim() == 5:
#             # [B, T, C, H, W]
#             B, T, C, H, W = x.shape
#             il = self.instance_length

#             if T % il != 0:
#                 if self.bag_size <= 0 or T % self.bag_size != 0:
#                     raise ValueError(
#                         f"Cannot infer instance_length/bag_size from input: "
#                         f"T={T}, instance_length={self.instance_length}, bag_size={self.bag_size}"
#                     )
#                 bag_size = self.bag_size
#                 il = T // bag_size
#             else:
#                 bag_size = T // il

#             x = rearrange(
#                 x,
#                 'b (t1 t2) c h w -> (b t1) c t2 h w',
#                 t1=bag_size,
#                 t2=il
#             )
#             bag_size_infer = bag_size

#         elif x.dim() == 6:
#             # [B, N, C, t, H, W]
#             B, N, C, t, H, W = x.shape
#             if N != self.bag_size:
#                 raise ValueError(
#                     f"Input bag_size N={N} != model.bag_size={self.bag_size}. "
#                     f"Check args.num_frames//args.instance_length and dataloader bag_size."
#                 )
#             x = rearrange(
#                 x,
#                 'b t1 c t2 h w -> (b t1) c t2 h w',
#                 t1=N,
#                 t2=t
#             )
#             bag_size_infer = N

#         else:
#             raise ValueError(f"Unexpected input dims: {x.dim()}, got shape={tuple(x.shape)}")

#         # backbone
#         x = self.features(x)
#         x = x.flatten(1)                                           # [(B*N), 512]

#         # reshape to [B, N, 512]
#         x = rearrange(x, '(b t) c -> b t c', t=bag_size_infer)

#         # ===== 提前应用 mask：在进入 MIL 之前就把被移除实例的 backbone 特征置零 =====
#         if instance_keep_mask is not None:
#             expected_shape = (x.shape[0], x.shape[1])  # [B, N]
#             if instance_keep_mask.shape != expected_shape:
#                 raise ValueError(
#                     f"instance_keep_mask shape {tuple(instance_keep_mask.shape)} "
#                     f"!= expected {expected_shape}"
#                 )
#             instance_keep_mask = instance_keep_mask.to(x.device).bool()
#             x = x * instance_keep_mask.unsqueeze(-1).float()

#         # MIL
#         if return_interpret:
#             inst_feat, attn_score, mil_aux = self.MIL(x, return_aux=True)
#         else:
#             inst_feat, attn_score = self.MIL(x, return_aux=False)
#             mil_aux = None

#         # baseline instance-level weights（用于解释展示）
#         baseline_weights = self.masked_softmax(attn_score, instance_keep_mask, dim=1)  # [B, N]

#         # ===== baseline path =====
#         if not self.use_dual_stream:
#             # 这里再乘一次 mask 属于保险做法，避免后续数值残留
#             if instance_keep_mask is not None:
#                 inst_feat_masked = inst_feat * instance_keep_mask.unsqueeze(-1).float()
#             else:
#                 inst_feat_masked = inst_feat

#             bag_feat = self.pwconv(inst_feat_masked).squeeze(1)    # [B, 1024]
#             out = self.fc(bag_feat)                                # [B, num_classes]

#             if out.dim() == 1:
#                 out = out.unsqueeze(1)

#             if return_interpret:
#                 aux = {
#                     "attn_score": attn_score.detach(),
#                     "baseline_weights": baseline_weights.detach(),
#                     "inst_feat": inst_feat.detach(),

#                     "bilstm_feat": mil_aux["bilstm_feat"],
#                     "mhsa_feat": mil_aux["mhsa_feat"],
#                     "asm_gate": mil_aux["asm_gate"],
#                     "attn_map": mil_aux["attn_map"],
#                 }
#                 return out, aux

#             return out

#         # ===== dual-stream path =====
#         # Stream-A: key prototype
#         key_logits = attn_score / self.score_tau
#         w_key = self.masked_softmax(key_logits, instance_keep_mask, dim=1)   # [B, N]
#         proto = (inst_feat * w_key.unsqueeze(-1)).sum(dim=1)                 # [B, 1024]
#         y_key = self.reg_key(proto)                                          # [B, num_classes]

#         # Stream-B: prototype-guided context
#         proto_norm = F.normalize(proto, dim=-1)                              # [B, 1024]
#         feat_norm = F.normalize(inst_feat, dim=-1)                           # [B, N, 1024]
#         sim = (feat_norm * proto_norm.unsqueeze(1)).sum(dim=-1)              # [B, N]
#         w_ctx = self.masked_softmax(sim, instance_keep_mask, dim=1)          # [B, N]
#         ctx = (inst_feat * w_ctx.unsqueeze(-1)).sum(dim=1)                   # [B, 1024]
#         y_ctx = self.reg_ctx(ctx)                                            # [B, num_classes]

#         # fusion
#         out = 0.5 * y_key + 0.5 * y_ctx

#         if out.dim() == 1:
#             out = out.unsqueeze(1)

#         if return_interpret:
#             aux = {
#                 "attn_score": attn_score.detach(),              # 原始实例分数
#                 "baseline_weights": baseline_weights.detach(),  # baseline实例权重（mask后）

#                 "key_weights": w_key.detach(),                  # key branch权重（mask后）
#                 "context_scores": sim.detach(),                 # context相似度
#                 "context_weights": w_ctx.detach(),              # context branch权重（mask后）

#                 "proto": proto.detach(),
#                 "ctx": ctx.detach(),
#                 "inst_feat": inst_feat.detach(),

#                 "bilstm_feat": mil_aux["bilstm_feat"],
#                 "mhsa_feat": mil_aux["mhsa_feat"],
#                 "asm_gate": mil_aux["asm_gate"],
#                 "attn_map": mil_aux["attn_map"],
#             }
#             return out, aux

#         return out




#为了画Grad-CAM图
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange

from utils import *   # 保留你工程里的 DMIN 等


class M3DFEL(nn.Module):
    def __init__(self, args):
        super(M3DFEL, self).__init__()

        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu'
        )

        # bag_size = num_frames // instance_length
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length

        # backbone
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-1])  # [B,512,1,1,1]

        # ===== Grad-CAM related =====
        self.gradcam_features = None
        self.gradcam_grads = None
        self._gradcam_hooks_registered = False
        self._gradcam_target_layer = None

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # MHSA
        self.heads = 8
        self.dim_head = 1024 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(1024, (self.dim_head * self.heads) * 3, bias=False)

        # stabilization / ASM-like gate
        self.norm = DMIN(num_features=1024)

        # baseline pooling
        self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
        self.fc = nn.Linear(1024, self.args.num_classes)

        # dual-stream settings
        self.use_dual_stream = getattr(self.args, "use_dual_stream", False)
        self.score_tau = float(getattr(self.args, "score_tau", 0.2))

        self.reg_key = nn.Linear(1024, self.args.num_classes)
        self.reg_ctx = nn.Linear(1024, self.args.num_classes)

        self.fuse_gate = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    # =========================
    # Grad-CAM hooks
    # =========================
    def _save_gradcam_features(self, module, inp, out):
        self.gradcam_features = out

    def _save_gradcam_grads(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradcam_grads = grad_output[0]

    def register_gradcam_hooks(self, target_layer_idx=7):
        """
        target_layer_idx:
            对于 self.features = Sequential(*list(r3d_18.children())[:-1])
            通常:
              0: stem
              1: layer1
              2: layer2
              3: layer3
              4: layer4
              ...
            但具体索引可能因 torchvision 版本略有不同。
            你可以先 print(self.features) 看一下。
        """
        if self._gradcam_hooks_registered:
            return

        if target_layer_idx < 0 or target_layer_idx >= len(self.features):
            raise ValueError(
                f"target_layer_idx={target_layer_idx} out of range for self.features with len={len(self.features)}"
            )

        self._gradcam_target_layer = self.features[target_layer_idx]
        self._gradcam_target_layer.register_forward_hook(self._save_gradcam_features)
        self._gradcam_target_layer.register_full_backward_hook(self._save_gradcam_grads)

        self._gradcam_hooks_registered = True

    def clear_gradcam_cache(self):
        self.gradcam_features = None
        self.gradcam_grads = None

    @staticmethod
    def masked_softmax(logits, keep_mask=None, dim=1):
        """
        logits:    [B, N]
        keep_mask: [B, N], True=保留, False=移除
        """
        if keep_mask is None:
            return torch.softmax(logits, dim=dim)

        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.bool()

        masked_logits = logits.masked_fill(~keep_mask, -1e9)
        return torch.softmax(masked_logits, dim=dim)

    def MIL(self, x, return_aux=False):
        """
        Inputs:
            x: [B, N, 512]

        Returns:
            inst_feat:  [B, N, 1024]
            attn_score: [B, N]

        If return_aux=True, also returns:
            bilstm_feat: [B, N, 1024]
            mhsa_feat:   [B, N, 1024]
            asm_gate:    [B, N, 1024]
            attn_map:    [B, N, N]
        """
        self.lstm.flatten_parameters()

        # 1) BiLSTM output
        bilstm_feat, _ = self.lstm(x)  # [B, N, 1024]

        # 2) MHSA
        qkv = self.to_qkv(bilstm_feat).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, H, N, N]
        attn = self.attend(dots)                                   # [B, H, N, N]

        mhsa_feat = torch.matmul(attn, v)                          # [B, H, N, Dh]
        mhsa_feat = rearrange(mhsa_feat, 'b h n d -> b n (h d)')   # [B, N, 1024]

        # 3) ASM-like gate
        asm_gate = self.norm(mhsa_feat)
        asm_gate = torch.sigmoid(asm_gate)                         # [B, N, 1024]

        # 4) enhanced instance feature
        inst_feat = bilstm_feat * asm_gate                         # [B, N, 1024]

        # 5) attention-derived instance score
        attn_map = attn.mean(dim=1)                                # [B, N, N]
        attn_score = attn_map.mean(dim=1)                          # [B, N]

        if return_aux:
            aux = {
                "bilstm_feat": bilstm_feat.detach(),
                "mhsa_feat": mhsa_feat.detach(),
                "asm_gate": asm_gate.detach(),
                "attn_map": attn_map.detach(),
            }
            return inst_feat, attn_score, aux

        return inst_feat, attn_score

    def forward(self, x, return_interpret=False, instance_keep_mask=None):
        """
        支持两种输入格式：
        1) x: [B, T, C, H, W]
        2) x: [B, N, C, t, H, W]

        instance_keep_mask:
            None 或 [B, N]
            True=保留, False=移除

        return_interpret:
            False -> 只返回 pred
            True  -> 返回 (pred, aux)
        """
        if x.dim() == 5:
            # [B, T, C, H, W]
            B, T, C, H, W = x.shape
            il = self.instance_length

            if T % il != 0:
                if self.bag_size <= 0 or T % self.bag_size != 0:
                    raise ValueError(
                        f"Cannot infer instance_length/bag_size from input: "
                        f"T={T}, instance_length={self.instance_length}, bag_size={self.bag_size}"
                    )
                bag_size = self.bag_size
                il = T // bag_size
            else:
                bag_size = T // il

            x = rearrange(
                x,
                'b (t1 t2) c h w -> (b t1) c t2 h w',
                t1=bag_size,
                t2=il
            )
            bag_size_infer = bag_size

        elif x.dim() == 6:
            # [B, N, C, t, H, W]
            B, N, C, t, H, W = x.shape
            if N != self.bag_size:
                raise ValueError(
                    f"Input bag_size N={N} != model.bag_size={self.bag_size}. "
                    f"Check args.num_frames//args.instance_length and dataloader bag_size."
                )
            x = rearrange(
                x,
                'b t1 c t2 h w -> (b t1) c t2 h w',
                t1=N,
                t2=t
            )
            bag_size_infer = N

        else:
            raise ValueError(f"Unexpected input dims: {x.dim()}, got shape={tuple(x.shape)}")

        # backbone
        x = self.features(x)                                       # [(B*N), 512, 1, 1, 1]
        x = x.flatten(1)                                           # [(B*N), 512]

        # reshape to [B, N, 512]
        x = rearrange(x, '(b t) c -> b t c', t=bag_size_infer)

        # 在进入 MIL 前应用 mask
        if instance_keep_mask is not None:
            expected_shape = (x.shape[0], x.shape[1])  # [B, N]
            if instance_keep_mask.shape != expected_shape:
                raise ValueError(
                    f"instance_keep_mask shape {tuple(instance_keep_mask.shape)} "
                    f"!= expected {expected_shape}"
                )
            instance_keep_mask = instance_keep_mask.to(x.device).bool()
            x = x * instance_keep_mask.unsqueeze(-1).float()

        # MIL
        if return_interpret:
            inst_feat, attn_score, mil_aux = self.MIL(x, return_aux=True)
        else:
            inst_feat, attn_score = self.MIL(x, return_aux=False)
            mil_aux = None

        # baseline instance-level weights（用于解释展示）
        baseline_weights = self.masked_softmax(attn_score, instance_keep_mask, dim=1)

        # ===== baseline path =====
        if not self.use_dual_stream:
            if instance_keep_mask is not None:
                inst_feat_masked = inst_feat * instance_keep_mask.unsqueeze(-1).float()
            else:
                inst_feat_masked = inst_feat

            bag_feat = self.pwconv(inst_feat_masked).squeeze(1)
            out = self.fc(bag_feat)

            if out.dim() == 1:
                out = out.unsqueeze(1)

            if return_interpret:
                aux = {
                    "attn_score": attn_score.detach(),
                    "baseline_weights": baseline_weights.detach(),
                    "inst_feat": inst_feat.detach(),

                    "bilstm_feat": mil_aux["bilstm_feat"],
                    "mhsa_feat": mil_aux["mhsa_feat"],
                    "asm_gate": mil_aux["asm_gate"],
                    "attn_map": mil_aux["attn_map"],
                }
                return out, aux

            return out

        # ===== dual-stream path =====
        # Stream-A: key prototype
        key_logits = attn_score / self.score_tau
        w_key = self.masked_softmax(key_logits, instance_keep_mask, dim=1)
        proto = (inst_feat * w_key.unsqueeze(-1)).sum(dim=1)
        y_key = self.reg_key(proto)

        # Stream-B: prototype-guided context
        proto_norm = F.normalize(proto, dim=-1)
        feat_norm = F.normalize(inst_feat, dim=-1)
        sim = (feat_norm * proto_norm.unsqueeze(1)).sum(dim=-1)
        w_ctx = self.masked_softmax(sim, instance_keep_mask, dim=1)
        ctx = (inst_feat * w_ctx.unsqueeze(-1)).sum(dim=1)
        y_ctx = self.reg_ctx(ctx)

        # fusion
        out = 0.5 * y_key + 0.5 * y_ctx

        if out.dim() == 1:
            out = out.unsqueeze(1)

        if return_interpret:
            aux = {
                "attn_score": attn_score.detach(),
                "baseline_weights": baseline_weights.detach(),

                "key_weights": w_key.detach(),
                "context_scores": sim.detach(),
                "context_weights": w_ctx.detach(),

                "proto": proto.detach(),
                "ctx": ctx.detach(),
                "inst_feat": inst_feat.detach(),

                "bilstm_feat": mil_aux["bilstm_feat"],
                "mhsa_feat": mil_aux["mhsa_feat"],
                "asm_gate": mil_aux["asm_gate"],
                "attn_map": mil_aux["attn_map"],
            }
            return out, aux

        return out




























# M3DFEL_avec.py
# 2026-02-23 Dual-stream(soft selection prototype anchor) version (AVEC2014 regression)
# Full replace file: model + profiling entry (Params / MACs / FLOPs)

# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchvision.models.video import r3d_18, R3D_18_Weights
# from einops import rearrange

# # 你的工程里必须有 DMIN
# from utils import *  # noqa


# class M3DFEL(nn.Module):
#     def __init__(self, args):
#         super(M3DFEL, self).__init__()

#         self.args = args
#         self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')

#         # bag_size = num_frames // instance_length
#         self.bag_size = self.args.num_frames // self.args.instance_length
#         self.instance_length = self.args.instance_length

#         # backbone networks
#         model = r3d_18(weights=R3D_18_Weights.DEFAULT)
#         self.features = nn.Sequential(*list(model.children())[:-1])  # output: [B,512,1,1,1]

#         # BiLSTM (instance contextualizer)
#         self.lstm = nn.LSTM(
#             input_size=512, hidden_size=512,
#             num_layers=2, batch_first=True, bidirectional=True
#         )

#         # multi-head self attention (MHSA)
#         self.heads = 8
#         self.dim_head = 1024 // self.heads
#         self.scale = self.dim_head ** -0.5
#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(1024, (self.dim_head * self.heads) * 3, bias=False)

#         # stabilization
#         self.norm = DMIN(num_features=1024)

#         # baseline pooling (depends on fixed bag_size=N)
#         self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
#         self.fc = nn.Linear(1024, self.args.num_classes)  # regression: num_classes=1

#         # ===== Dual-stream settings =====
#         self.use_dual_stream = getattr(self.args, "use_dual_stream", False)

#         # soft selection scaling factor (smaller => sharper/sparser, larger => smoother)
#         self.score_tau = float(getattr(self.args, "score_tau", 0.2))

#         # Stream-A / Stream-B regressors
#         self.reg_key = nn.Linear(1024, self.args.num_classes)
#         self.reg_ctx = nn.Linear(1024, self.args.num_classes)

#         # (optional) gate fusion if you want later; default not used
#         self.fuse_gate = nn.Sequential(
#             nn.Linear(2048, 1),
#             nn.Sigmoid()
#         )

#     def MIL(self, x):
#         """
#         Inputs:
#             x: [B, N, 512]
#         Returns:
#             inst_feat: [B, N, 1024]  (contextual instance features)
#             attn_score: [B, N]       (instance importance score derived from MHSA attention)
#         """
#         self.lstm.flatten_parameters()
#         x, _ = self.lstm(x)  # [B, N, 1024]
#         ori_x = x

#         # MHSA
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, H, N, N]
#         attn = self.attend(dots)                                   # [B, H, N, N]
#         x = torch.matmul(attn, v)                                  # [B, H, N, Dh]
#         x = rearrange(x, 'b h n d -> b n (h d)')                    # [B, N, 1024]

#         # stabilization gate
#         x = self.norm(x)
#         x = torch.sigmoid(x)
#         inst_feat = ori_x * x                                       # [B, N, 1024]

#         # derive instance score from attention:
#         # mean over heads, mean over query dimension -> each key's average attention mass
#         # attn: [B,H,N,N] -> [B,N]
#         attn_score = attn.mean(dim=1).mean(dim=1)

#         return inst_feat, attn_score

#     def forward(self, x):
#         """
#         支持两种输入格式：
#         1) x: [B, num_frames, C, H, W]    (num_frames = bag_size * instance_length)
#         2) x: [B, bag_size, C, il, H, W] (MIL dataloader 输出)

#         最终统一成 [(B*bag_size), C, il, H, W] 送入3D backbone
#         """
#         if x.dim() == 5:
#             # x: [B, T, C, H, W]
#             B, T, C, H, W = x.shape
#             il = self.instance_length

#             if T % il != 0:
#                 # fallback infer il using self.bag_size
#                 if self.bag_size <= 0 or T % self.bag_size != 0:
#                     raise ValueError(
#                         f"Cannot infer instance_length/bag_size from input: T={T}, "
#                         f"instance_length={self.instance_length}, bag_size={self.bag_size}"
#                     )
#                 bag_size = self.bag_size
#                 il = T // bag_size
#             else:
#                 bag_size = T // il

#             x = rearrange(
#                 x,
#                 'b (t1 t2) c h w -> (b t1) c t2 h w',
#                 t1=bag_size,
#                 t2=il
#             )
#             bag_size_infer = bag_size

#         elif x.dim() == 6:
#             # x: [B, N, C, t, H, W]
#             B, N, C, t, H, W = x.shape
#             if N != self.bag_size:
#                 raise ValueError(
#                     f"Input bag_size N={N} != model.bag_size={self.bag_size}. "
#                     f"Check args.num_frames//args.instance_length and dataloader bag_size."
#                 )
#             x = rearrange(
#                 x,
#                 'b t1 c t2 h w -> (b t1) c t2 h w',
#                 t1=N,
#                 t2=t
#             )
#             bag_size_infer = N

#         else:
#             raise ValueError(f"Unexpected input dims: {x.dim()}, expected 5 or 6. Got shape={tuple(x.shape)}")

#         # backbone: -> [B*bag, 512, 1, 1, 1] -> flatten -> [B*bag, 512]
#         x = self.features(x)
#         x = x.flatten(1)

#         # -> [B, N, 512]
#         x = rearrange(x, '(b t) c -> b t c', t=bag_size_infer)

#         # -> [B, N, 1024] and attention-based scores [B, N]
#         inst_feat, attn_score = self.MIL(x)

#         # ===== baseline path (single-stream pooling) =====
#         if not self.use_dual_stream:
#             bag_feat = self.pwconv(inst_feat).squeeze(1)  # [B, 1024]
#             out = self.fc(bag_feat)                       # [B, 1]
#             if out.dim() == 1:
#                 out = out.unsqueeze(1)
#             return out

#         # ===== dual-stream path (ours) =====
#         # Stream-A: soft selection prototype (differentiable)
#         w_key = torch.softmax(attn_score / self.score_tau, dim=1)    # [B,N]
#         proto = (inst_feat * w_key.unsqueeze(-1)).sum(dim=1)         # [B,1024]
#         y_key = self.reg_key(proto)                                  # [B,1]

#         # Stream-B: prototype-guided context pooling (cosine similarity)
#         proto_norm = F.normalize(proto, dim=-1)                      # [B,1024]
#         feat_norm = F.normalize(inst_feat, dim=-1)                   # [B,N,1024]
#         sim = (feat_norm * proto_norm.unsqueeze(1)).sum(dim=-1)      # [B,N]
#         w_ctx = torch.softmax(sim, dim=1)                            # [B,N]
#         ctx = (inst_feat * w_ctx.unsqueeze(-1)).sum(dim=1)           # [B,1024]
#         y_ctx = self.reg_ctx(ctx)                                    # [B,1]

#         # Fusion: parameter-free average
#         out = 0.5 * y_key + 0.5 * y_ctx

#         if out.dim() == 1:
#             out = out.unsqueeze(1)
#         return out


# # ===========================
# # Profiling entry (Params + MACs/FLOPs)
# # ===========================
# if __name__ == "__main__":
#     from types import SimpleNamespace

#     try:
#         from thop import profile
#     except Exception as e:
#         raise RuntimeError(
#             "thop is not installed in this python environment. "
#             "Run:  python -m pip install thop"
#         ) from e

#     def count_params(model):
#         total = sum(p.numel() for p in model.parameters())
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         return total, trainable

#     @torch.no_grad()
#     def run_profile(use_dual: bool, num_frames: int, instance_length: int, h: int, w: int, tau: float):
#         args = SimpleNamespace(
#             gpu_ids=[0],
#             num_frames=num_frames,
#             instance_length=instance_length,
#             num_classes=1,             # AVEC regression
#             use_dual_stream=use_dual,
#             score_tau=tau,
#         )
#         model = M3DFEL(args).cuda().eval()

#         # dummy: [B, N, C, il, H, W]  (最稳的输入形式)
#         B = 1
#         N = num_frames // instance_length
#         C = 3
#         il = instance_length
#         dummy = torch.randn(B, N, C, il, h, w).cuda()

#         total, trainable = count_params(model)

#         macs, params_thop = profile(model, inputs=(dummy,), verbose=False)
#         flops = 2 * macs  # 口径：FLOPs ≈ 2 * MACs（论文里写清楚）

#         tag = "OURS (dual-stream)" if use_dual else "BASELINE (single-stream)"
#         print("\n" + "=" * 70)
#         print(f"[{tag}] input = [B={B}, N={N}, C=3, il={il}, H={h}, W={w}]")
#         print(f"Params(total): {total/1e6:.3f} M | trainable: {trainable/1e6:.3f} M")
#         print(f"THOP Params : {params_thop/1e6:.3f} M")
#         print(f"MACs        : {macs/1e9:.3f} G")
#         print(f"FLOPs(2*MACs): {flops/1e9:.3f} G")

#     # ====== 你按真实训练配置改这四个数 ======
#     NUM_FRAMES = 64          # 你训练的 args.num_frames
#     INSTANCE_LEN = 4         # 你训练的 args.instance_length
#     H = 224                  # 你训练的分辨率（112 or 224）
#     W = 224
#     TAU = 0.2                # 你常用的 score_tau

#     # 先跑 baseline，再跑 ours（方便你写“增量开销”）
#     run_profile(use_dual=False, num_frames=NUM_FRAMES, instance_length=INSTANCE_LEN, h=H, w=W, tau=TAU)
#     run_profile(use_dual=True,  num_frames=NUM_FRAMES, instance_length=INSTANCE_LEN, h=H, w=W, tau=TAU)