


import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from math import ceil
import pandas as pd

from . import transforms3d  # 你的 transforms3d.py（确保可 import）


# ========= Kinetics / video pretrain 常用 mean/std（对应 [0,1] 输入）=========
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD  = (0.22803, 0.22145, 0.216989)


#IMAGENET的normalize
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# =========================
# 3D / video transforms (sequence-consistent)
# 输入约定：numpy array, shape = (C, T, H, W), dtype float32, range [0,255]
# 输出：同 shape 同 dtype
# =========================
class Compose3D:
    def __init__(self, transforms):
        self.transforms = list(transforms) if transforms is not None else []

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RandomHorizontalFlip3D:
    """对所有帧做同一次随机水平翻转（序列一致）"""
    def __init__(self, p=0.5):
        self.p = float(p)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (C,T,H,W)
        if np.random.rand() < self.p:
            # flip W axis
            return x[..., ::-1].copy()
        return x


class ColorJitter3D:
    """
    对所有帧用同一组 jitter 参数（序列一致）。
    仅依赖 PIL.ImageEnhance，避免引入 torchvision。
    """
    def __init__(self, strength=0.4):
        self.strength = float(strength)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.strength <= 0:
            return x

        # 采样一次参数，应用到所有帧
        # brightness/contrast/saturation: factor in [1-s, 1+s]
        s = self.strength
        b = np.random.uniform(1 - s, 1 + s)
        c = np.random.uniform(1 - s, 1 + s)
        sat = np.random.uniform(1 - s, 1 + s)

        C, T, H, W = x.shape
        # 转为 (T,H,W,C)
        frames = np.transpose(x, (1, 2, 3, 0))  # float32 [0,255]
        out = np.empty_like(frames, dtype=np.float32)

        for t in range(T):
            im = frames[t]
            im_u8 = np.clip(im, 0, 255).astype(np.uint8)
            pil = Image.fromarray(im_u8)

            pil = ImageEnhance.Brightness(pil).enhance(b)
            pil = ImageEnhance.Contrast(pil).enhance(c)
            pil = ImageEnhance.Color(pil).enhance(sat)

            out[t] = np.asarray(pil, dtype=np.float32)

        # 回到 (C,T,H,W)
        out = np.transpose(out, (3, 0, 1, 2))
        return out


def build_avec_transform(is_train: bool, color_jitter: float = 0.4, hflip_p: float = 0.5):
    """
    供你在外部创建 transform pipeline 用：
    - train: flip + color jitter（都序列一致）
    - val/test: 默认不做随机增强（返回 None 或空 Compose3D）
    """
    if not is_train:
        # 验证/测试一般不做随机增强
        return None

    return Compose3D([
        RandomHorizontalFlip3D(p=hflip_p),
        ColorJitter3D(strength=color_jitter),
    ])


def _numeric_sort_key(filename: str) -> int:
    """从文件名提取开头数字做排序key：000123.jpg -> 123"""
    m = re.match(r"(\d+)", filename)
    return int(m.group(1)) if m else 10**18


def _list_frames_sorted(frame_dir: str, use_numeric_sort: bool = True):
    """返回帧文件名列表（只返回文件名），按时间顺序排好"""
    names = [n for n in os.listdir(frame_dir) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    if use_numeric_sort:
        names.sort(key=_numeric_sort_key)
    else:
        names = sorted(names, key=str)
    return names


def _build_frame_dir(root: str, split: str, task: str, sample_id: str) -> str:
    """
    按你的数据组织方式拼路径：
    {root}/{split}/{task}/{sample_id}/
    例：avec14_processed/train/Freeform/203_1/
    """
    return os.path.join(root, split, task, sample_id)


def _read_label_csv(csv_path: str):
    """
    读取 label.csv，格式：
    file,label
    203_1,3
    """
    df = pd.read_csv(csv_path)
    if "file" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must contain columns ['file','label'], got: {df.columns.tolist()}")

    sample_ids = df["file"].astype(str).tolist()
    labels = df["label"].astype(float).tolist()

    # 防呆：万一有人写成 203_1.mp4
    sample_ids = [sid.replace(".mp4", "").replace(".avi", "").replace(".mov", "") for sid in sample_ids]
    return sample_ids, labels


def _to_bag_instances(image_cthw: torch.Tensor, bag_size: int) -> torch.Tensor:
    """
    输入:  image (C,T,H,W)
    输出:  bag (N,C,t,H,W) 其中 N=bag_size, t=T/N
    """
    assert image_cthw.dim() == 4, f"expect (C,T,H,W), got {tuple(image_cthw.shape)}"
    C, T, H, W = image_cthw.shape
    assert T % bag_size == 0, f"T({T}) must be divisible by bag_size({bag_size})"
    t = T // bag_size
    bag = image_cthw.view(C, bag_size, t, H, W).permute(1, 0, 2, 3, 4).contiguous()
    return bag


class AVECTrainMIL(Dataset):
    """
    训练集：
    - 从长视频帧文件夹中：在(下采样后的序列上)随机选择 K 个连续窗口 frame_len（默认64）
    - 每个窗口组成一个 bag
    - bag 内按 bag_size 切成 N 个 instance（例如 N=16 -> 每个instance=4帧）
    输出: (K,N,C,t,H,W), label
    """

    def __init__(
        self,
        root: str,
        split: str,            # 'train'
        task: str,             # 'Freeform'
        label_csv: str,        # '.../train_label.csv'
        frame_len: int = 64,
        img_size: int = 224,
        input_channel: int = 3,
        sample_interval: int = 3,
        transform=None,
        bag_size: int = 16,
        train_num_segs: int = 1,         # ✅ 新增：K
        use_numeric_sort: bool = True
    ):
        self.root = root
        self.split = split
        self.task = task

        self.frame_len = frame_len
        self.img_size = img_size
        self.input_channel = input_channel
        self.sample_interval = sample_interval
        self.transform = transform
        self.bag_size = bag_size
        self.train_num_segs = int(train_num_segs)
        self.use_numeric_sort = use_numeric_sort

        assert self.train_num_segs >= 1, f"train_num_segs must be >= 1, got {self.train_num_segs}"
        assert frame_len % bag_size == 0, f"frame_len({frame_len}) must be divisible by bag_size({bag_size})"

        self.sample_ids, self.labels = _read_label_csv(label_csv)

    def __len__(self):
        return len(self.sample_ids)

    def _load_window_as_bag(self, frame_dir: str, names: list, frames: list):
        """
        读取一个 window（长度 frame_len）并转为 bag: (N,C,t,H,W)
        """
        # 读帧 -> (C,T,H,W) 的 numpy
        image_pack = np.empty((self.frame_len, self.img_size, self.img_size, 3), dtype=np.float32)
        for i, fidx in enumerate(frames):
            im = Image.open(os.path.join(frame_dir, names[fidx])).convert("RGB")
            im = im.resize((self.img_size, self.img_size))
            image_pack[i] = np.asarray(im, dtype=np.float32)  # 不在这里 /255（to_tensor 内部会做）

        image = np.transpose(image_pack, (3, 0, 1, 2))  # (C,T,H,W)

        if self.transform is not None:
            image = self.transform(image)

        if self.input_channel == 1:
            image = transforms3d.rgb_to_gray(image)

        # to_tensor() 内部已经做了 /255 -> [0,1]
        image = transforms3d.to_tensor(image)

        # ✅显式使用 Kinetics mean/std（不用改 transforms3d.py）
        # image = transforms3d.normalize(image, mean=KINETICS_MEAN, std=KINETICS_STD)
        image = transforms3d.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)   #2026年2月25日改
        bag = _to_bag_instances(image, self.bag_size)  # (N,C,t,H,W)
        return bag

    def __getitem__(self, idx):
        label = self.labels[idx]
        sample_id = self.sample_ids[idx]

        frame_dir = _build_frame_dir(self.root, self.split, self.task, sample_id)
        names = _list_frames_sorted(frame_dir, use_numeric_sort=self.use_numeric_sort)

        # 下采样
        names = names[:: self.sample_interval]
        L = len(names)

        K = self.train_num_segs

        # 在下采样序列上采 K 个窗口
        if L <= self.frame_len:
            # 不足一段：用均匀采样补齐（K 段都会相同，属于合理退化）
            base_frames = [(i * L // self.frame_len) for i in range(self.frame_len)]
            bags = [self._load_window_as_bag(frame_dir, names, base_frames) for _ in range(K)]
        else:
            max_start = L - self.frame_len
            # 尽量不重复 start；若可选数不足则允许重复
            replace = (max_start + 1) < K
            starts = np.random.choice(np.arange(0, max_start + 1), size=K, replace=replace).tolist()
            bags = []
            for s in starts:
                frames = list(range(int(s), int(s) + self.frame_len))
                bags.append(self._load_window_as_bag(frame_dir, names, frames))

        # stack -> (K,N,C,t,H,W)
        bags = torch.stack(bags, dim=0)
        return bags, float(label)


class AVECValTestMIL(Dataset):
    """
    验证/测试集：
    - 把整段视频切成多个连续片段（每段 frame_len=64）
    - 每段片段作为一个 bag，再切成 N 个 instance
    输出: (S,N,C,t,H,W), label
    """

    def __init__(
        self,
        root: str,
        split: str,            # 'dev' 或 'test'
        task: str,
        label_csv: str,
        frame_len: int = 64,
        img_size: int = 224,
        input_channel: int = 3,
        sample_interval: int = 3,
        transform=None,
        bag_size: int = 16,
        use_numeric_sort: bool = True
    ):
        self.root = root
        self.split = split
        self.task = task

        self.frame_len = frame_len
        self.img_size = img_size
        self.input_channel = input_channel
        self.sample_interval = sample_interval
        self.transform = transform
        self.bag_size = bag_size
        self.use_numeric_sort = use_numeric_sort

        assert frame_len % bag_size == 0, f"frame_len({frame_len}) must be divisible by bag_size({bag_size})"

        self.sample_ids, self.labels = _read_label_csv(label_csv)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sample_id = self.sample_ids[idx]

        frame_dir = _build_frame_dir(self.root, self.split, self.task, sample_id)
        names = _list_frames_sorted(frame_dir, use_numeric_sort=self.use_numeric_sort)

        # 下采样
        names = names[:: self.sample_interval]
        L = len(names)

        # 生成多个连续段起点（覆盖整段视频）
        frames_list = []
        if L <= self.frame_len:
            one = [(i * L // self.frame_len) for i in range(self.frame_len)]
            frames_list.append(one)
        else:
            section = ceil(L / self.frame_len)
            if section == 1:
                starts = [0]
            else:
                excess = section * self.frame_len - L
                shrink_per_gap = excess / (section - 1)
                step = self.frame_len - shrink_per_gap
                starts = [int(i * step) for i in range(section)]
                starts = [min(s, L - self.frame_len) for s in starts]
                starts = sorted(set(starts))

            for s in starts:
                frames_list.append(list(range(s, s + self.frame_len)))

        # 输出： (S,N,C,t,H,W)
        N = self.bag_size
        t = self.frame_len // N
        C_out = 1 if self.input_channel == 1 else 3
        S = len(frames_list)
        out = torch.empty((S, N, C_out, t, self.img_size, self.img_size), dtype=torch.float32)

        image_pack = np.empty((self.frame_len, self.img_size, self.img_size, 3), dtype=np.float32)

        for s_idx, frames in enumerate(frames_list):
            for i, fidx in enumerate(frames):
                im = Image.open(os.path.join(frame_dir, names[fidx])).convert("RGB")
                im = im.resize((self.img_size, self.img_size))
                image_pack[i] = np.asarray(im, dtype=np.float32)  # 不在这里 /255

            image = np.transpose(image_pack, (3, 0, 1, 2))  # (C,T,H,W)

            if self.transform is not None:
                image = self.transform(image)

            if self.input_channel == 1:
                image = transforms3d.rgb_to_gray(image)

            image = transforms3d.to_tensor(image)  # 内部 /255
            image = transforms3d.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD) 
            # image = transforms3d.normalize(image, mean=KINETICS_MEAN, std=KINETICS_STD)

            out[s_idx] = _to_bag_instances(image, self.bag_size)

        return out, float(label),sample_id



