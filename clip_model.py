"""Self-contained CLIP wrapper (open_clip) — image AND text encoding, L2-normalized."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np


_DTYPES = {"fp32": None, "fp16": "float16", "bf16": "bfloat16"}


class ClipModel:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None,
                 precision="fp32"):
        import torch, open_clip
        # Disable cuDNN: ViT-B-32 is almost all matmul; its single patch-embed conv
        # otherwise needs a cuDNN workspace that fails ("unable to find an engine")
        # when many processes share a busy GPU. Native conv needs no workspace.
        torch.backends.cudnn.enabled = False
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if precision not in _DTYPES:
            raise ValueError(f"precision must be one of {list(_DTYPES)}, got {precision!r}")
        self.dtype = getattr(torch, _DTYPES[precision]) if _DTYPES[precision] else torch.float32
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.to(self.device, dtype=self.dtype).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._gpu_pp = None  # built lazily by encode_image_files
        try:
            tp = getattr(self.model, "text_projection", None)
            self.dim = int(tp.shape[1]) if hasattr(tp, "shape") else int(getattr(tp, "out_features", 512))
        except Exception:
            self.dim = 512

    def encode_images(self, pil_images: list, batch_size=64) -> np.ndarray:
        torch = self.torch
        feats = []
        for i in range(0, len(pil_images), batch_size):
            batch = [self.preprocess(im) for im in pil_images[i:i + batch_size]]
            with torch.no_grad():
                x = torch.stack(batch).to(self.device, dtype=self.dtype)
                f = self.model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.float().cpu().numpy().astype(np.float16))
        return np.concatenate(feats, 0) if feats else np.zeros((0, self.dim), np.float16)

    def encode_image_files(self, paths: list, batch_size=64) -> np.ndarray:
        """Same output as encode_images, but decodes/resizes on the GPU.

        Takes file paths rather than PIL images: the decode happens in nvjpeg, so
        handing it a decoded PIL image would defeat the point.
        """
        torch = self.torch
        if self._gpu_pp is None:
            from gpu_preprocess import GpuPreprocess
            self._gpu_pp = GpuPreprocess(self.model, self.device, self.dtype)
        feats = []
        for i in range(0, len(paths), batch_size):
            with torch.no_grad():
                x = self._gpu_pp(paths[i:i + batch_size])
                f = self.model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.float().cpu().numpy().astype(np.float16))
        return np.concatenate(feats, 0) if feats else np.zeros((0, self.dim), np.float16)

    def encode_texts(self, texts: list[str], batch_size=256) -> np.ndarray:
        torch = self.torch
        feats = []
        for i in range(0, len(texts), batch_size):
            toks = self.tokenizer(texts[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                f = self.model.encode_text(toks)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.float().cpu().numpy().astype(np.float32))
        return np.concatenate(feats, 0) if feats else np.zeros((0, self.dim), np.float32)
