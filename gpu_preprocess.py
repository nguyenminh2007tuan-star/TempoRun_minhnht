"""GPU-side JPEG decode + resize + normalize, mirroring open_clip's CPU transform.

Why this exists: on a shared box the CPU is the scarce resource (dozens of jobs
competing for cores) while the GPU sits idle waiting for input — the PIL decode +
resize in open_clip's CPU transform starves the model. nvjpeg decodes on the GPU
and the resize/normalize are plain tensor ops, so the whole input path moves off
the CPU; only the raw file read stays.

The transform is rebuilt from the model's own `preprocess_cfg` (size, resize mode,
interpolation, mean/std) so each encoder keeps its own preprocessing — SigLIP
squashes to a square, CLIP-style models resize the short side then centre-crop,
and the normalisation constants differ per family.
"""
from __future__ import annotations


class GpuPreprocess:
    def __init__(self, model, device, dtype):
        import torch
        import open_clip
        self.torch = torch
        self.device = device
        self.dtype = dtype

        cfg = open_clip.get_model_preprocess_cfg(model)
        size = cfg["size"]
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.resize_mode = cfg.get("resize_mode", "shortest")
        interp = str(cfg.get("interpolation", "bicubic"))
        self.interp = "bilinear" if interp == "bilinear" else "bicubic"
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    def _read(self, path):
        import numpy as np
        torch = self.torch
        with open(path, "rb") as fh:
            buf = np.frombuffer(fh.read(), dtype=np.uint8)
        return torch.from_numpy(buf.copy())  # copy(): frombuffer is read-only

    def _resize(self, x, h, w):
        """[N,3,h,w] float -> [N,3,size] following the model's resize_mode."""
        import torch.nn.functional as F
        th, tw = self.size
        if self.resize_mode == "squash":
            return F.interpolate(x, size=(th, tw), mode=self.interp,
                                 align_corners=False, antialias=True)
        # "shortest": scale so both sides cover the target, then centre-crop
        scale = max(th / h, tw / w)
        nh, nw = max(th, int(round(h * scale))), max(tw, int(round(w * scale)))
        x = F.interpolate(x, size=(nh, nw), mode=self.interp,
                          align_corners=False, antialias=True)
        top, left = (nh - th) // 2, (nw - tw) // 2
        return x[:, :, top:top + th, left:left + tw]

    def __call__(self, paths: list) -> "object":
        """Paths -> normalized float batch [N,3,H,W] on device, ready for encode_image."""
        import torch
        from torchvision.io import decode_jpeg
        import torch.nn.functional as F

        raw = [self._read(p) for p in paths]
        imgs = decode_jpeg(raw, device=self.device)  # list of uint8 [3,H,W] on GPU

        # Resize per distinct source shape rather than per image: frames from one
        # video share a resolution, so this is normally a single batched kernel
        # instead of one launch per frame — which matters when the CPU issuing
        # them is contended.
        groups = {}
        for i, im in enumerate(imgs):
            groups.setdefault(tuple(im.shape[-2:]), []).append(i)

        out = [None] * len(imgs)
        for (h, w), idxs in groups.items():
            x = torch.stack([imgs[i] for i in idxs]).float()
            x = self._resize(x, h, w)
            for j, i in enumerate(idxs):
                out[i] = x[j]

        batch = torch.stack(out, 0).clamp_(0, 255).div_(255.0)
        batch = (batch - self.mean) / self.std
        return batch.to(dtype=self.dtype)
