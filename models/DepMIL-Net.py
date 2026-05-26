
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange

from utils import *  


class DepMIL_Net(nn.Module):
    def __init__(self, args):
        super(DepMIL_Net, self).__init__()

        self.args = args
        self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')

        # bag_size = num_frames // instance_length
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length

        # backbone networks
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-1])  # output: [B,512,1,1,1]

        # BiLSTM (instance contextualizer)
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=512,
            num_layers=2, batch_first=True, bidirectional=True
        )

        # multi-head self attention (MHSA)
        self.heads = 8
        self.dim_head = 1024 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(1024, (self.dim_head * self.heads) * 3, bias=False)

        # stabilization
        self.norm = DMIN(num_features=1024)

        # baseline pooling (depends on fixed bag_size=N)
        self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)
        self.fc = nn.Linear(1024, self.args.num_classes)  # regression: num_classes=1

        # ===== Dual-stream settings =====
        self.use_dual_stream = getattr(self.args, "use_dual_stream", False)

        # soft selection temperature (smaller => sparser, larger => smoother)
        self.score_tau = float(getattr(self.args, "score_tau", 0.2))

        # Stream-A / Stream-B regressors
        self.reg_key = nn.Linear(1024, self.args.num_classes)
        self.reg_ctx = nn.Linear(1024, self.args.num_classes)

        # (optional) gate fusion if you want later; default not used
        self.fuse_gate = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def MIL(self, x):
        """
        Inputs:
            x: [B, N, 512]
        Returns:
            inst_feat: [B, N, 1024]  (contextual instance features)
            attn_score: [B, N]       (instance importance score derived from MHSA attention)
        """
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # [B, N, 1024]
        ori_x = x

        # MHSA
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, H, N, N]
        attn = self.attend(dots)                                   # [B, H, N, N]
        x = torch.matmul(attn, v)                                  # [B, H, N, Dh]
        x = rearrange(x, 'b h n d -> b n (h d)')                    # [B, N, 1024]

        # stabilization gate
        x = self.norm(x)
        x = torch.sigmoid(x)
        inst_feat = ori_x * x                                       # [B, N, 1024]

        # derive instance score from attention:
        # mean over heads, mean over query dimension -> each key's average attention mass
        # attn: [B,H,N,N] -> [B,N]
        attn_score = attn.mean(dim=1).mean(dim=1)

        return inst_feat, attn_score

    def forward(self, x):
        """
        支持两种输入格式：
        1) x: [B, num_frames, C, H, W]    (num_frames = bag_size * instance_length)
        2) x: [B, bag_size, C, il, H, W] (MIL dataloader 输出)

        最终统一成 [(B*bag_size), C, il, H, W] 送入3D backbone
        """
        if x.dim() == 5:
            # x: [B, T, C, H, W]
            B, T, C, H, W = x.shape
            il = self.instance_length

            if T % il != 0:
                # fallback infer il using self.bag_size
                if self.bag_size <= 0 or T % self.bag_size != 0:
                    raise ValueError(
                        f"Cannot infer instance_length/bag_size from input: T={T}, "
                        f"instance_length={self.instance_length}, bag_size={self.bag_size}"
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
            # x: [B, N, C, t, H, W]
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
            raise ValueError(f"Unexpected input dims: {x.dim()}, expected 5 or 6. Got shape={tuple(x.shape)}")

        # backbone: -> [B*bag, 512, 1, 1, 1] -> flatten -> [B*bag, 512]
        x = self.features(x)
        x = x.flatten(1)

        # -> [B, N, 512]
        x = rearrange(x, '(b t) c -> b t c', t=bag_size_infer)

        # -> [B, N, 1024] and attention-based scores [B, N]
        inst_feat, attn_score = self.MIL(x)

        # ===== baseline path (single-stream pooling) =====
        if not self.use_dual_stream:
            bag_feat = self.pwconv(inst_feat).squeeze(1)  # [B, 1024]
            out = self.fc(bag_feat)                       # [B, 1]
            if out.dim() == 1:
                out = out.unsqueeze(1)
            return out

        # ===== dual-stream path (ours) =====
        # Stream-A: soft selection prototype (differentiable)
        # w_key: [B,N]
        w_key = torch.softmax(attn_score / self.score_tau, dim=1)
        proto = (inst_feat * w_key.unsqueeze(-1)).sum(dim=1)         # [B, 1024]
        y_key = self.reg_key(proto)                                  # [B, 1]

        # Stream-B: prototype-guided context pooling
        proto_norm = F.normalize(proto, dim=-1)                      # [B,1024]
        feat_norm = F.normalize(inst_feat, dim=-1)                   # [B,N,1024]
        sim = (feat_norm * proto_norm.unsqueeze(1)).sum(dim=-1)      # [B,N]
        w_ctx = torch.softmax(sim, dim=1)                            # [B,N]
        ctx = (inst_feat * w_ctx.unsqueeze(-1)).sum(dim=1)           # [B,1024]
        y_ctx = self.reg_ctx(ctx)                                    # [B,1]

        # Fusion: stable parameter-free average (recommended first)
        out = 0.5 * y_key + 0.5 * y_ctx

        # If you later want gate fusion, replace the above with:
        # gate = self.fuse_gate(torch.cat([proto, ctx], dim=-1))      # [B,1]
        # out = gate * y_key + (1 - gate) * y_ctx

        if out.dim() == 1:
            out = out.unsqueeze(1)

        return out


