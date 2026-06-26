"""Self-contained CLIP wrapper (open_clip) — image AND text encoding, L2-normalized."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np


class ClipModel:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None):
        import torch, open_clip
        # Disable cuDNN: ViT-B-32 is almost all matmul; its single patch-embed conv
        # otherwise needs a cuDNN workspace that fails ("unable to find an engine")
        # when many processes share a busy GPU. Native conv needs no workspace.
        torch.backends.cudnn.enabled = False
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
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
                x = torch.stack(batch).to(self.device)
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
