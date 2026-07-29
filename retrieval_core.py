"""Shared retrieval primitives used by both retrieve.py (single encoder) and
retrieve_fusion.py (multi-encoder RRF). Kept separate so both scripts run the
exact same chunked top-K GPU search instead of two copies drifting apart.
"""
from __future__ import annotations
import glob, os, re
from pathlib import Path
import numpy as np

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Heuristic, no-LLM query decomposition: split on sentence boundaries.

    Public-round queries average ~71 CLIP-BPE tokens (21% exceed the 77-token
    limit; SigLIP2's 64-token limit truncates 76%) because each query narrates
    several shots back to back ("... The scene shifts to ... then to ...").
    Splitting on sentences alone (measured on public_round_tasks.jsonl) drops
    truncation to ~0.3%/1.7% respectively, at ~2.7 sub-queries/task on average
    — cheap enough to not need an LLM for this.
    """
    parts = [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def load_index(shard_dir):
    """Load every <video_id>.npz shard into one flat keyframe index."""
    embs, vids, ts = [], [], []
    files = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    for f in files:
        d = np.load(f)
        e = d["emb"]
        if e.shape[0] == 0:
            continue
        embs.append(e)
        vid = Path(f).stem
        vids.extend([vid] * e.shape[0])
        ts.append(d["ts_ms"])
    if not embs:
        raise SystemExit(f"no shards in {shard_dir}")
    emb = np.concatenate(embs, 0)            # [N, D] fp16
    ts = np.concatenate(ts, 0).astype(np.int32)
    vids = np.array(vids)
    print(f"[index] {emb.shape[0]} keyframes from {len(files)} videos, dim={emb.shape[1]}", flush=True)
    return emb, vids, ts


def top_frames(scored, top_k=10, min_gap_ms=0):
    """Pick the final ≤top_k frames from scored candidates, ranked purely by score.

    scored: iterable of (video_id, frame_ms, score). No dedup-to-distinct-video —
    each of a task's ≤10 predictions is just one (video_id, frame_ms), and the metric
    only rewards the reciprocal rank of the first in-interval hit, so the best move is
    to rank frames by score and let each frame carry its own video_id. The frame/video
    mix then falls out of the scores themselves (a confident query concentrates on one
    video, an unsure one spreads) instead of being forced by a rule.

    min_gap_ms > 0 applies a light temporal NMS: skip a frame within min_gap_ms of an
    already-picked frame FROM THE SAME VIDEO, so dense (≤2s) near-duplicate keyframes
    of one moment don't eat several of the 10 slots. min_gap_ms=0 = pure score order.
    """
    ranked = sorted(scored, key=lambda x: x[2], reverse=True)
    picked, kept_by_vid = [], {}
    for vid, fms, score in ranked:
        if min_gap_ms > 0:
            if any(abs(fms - k) < min_gap_ms for k in kept_by_vid.get(vid, ())):
                continue
            kept_by_vid.setdefault(vid, []).append(fms)
        picked.append((vid, fms, score))
        if len(picked) >= top_k:
            break
    return picked


def chunked_topk(Qt, idx, K, chunk=200_000):
    """Cosine top-K per query, chunked over the keyframe index to bound GPU memory.

    Qt:  [T, D] half tensor of L2-normalized query embeddings (on the compute device)
    idx: [N, D] half tensor of L2-normalized keyframe embeddings. May live on CPU:
         if it is not on Qt's device, each chunk is copied over on the fly so the
         full index is never resident on the GPU (survives a shared, contended card).
    Returns (top_val [T,K] float16, top_idx [T,K] long) — indices into idx's rows.
    """
    import torch
    dev = Qt.device
    stream = idx.device != dev
    T, N = Qt.shape[0], idx.shape[0]
    K = min(K, N)
    top_val = torch.full((T, K), float("-inf"), device=dev, dtype=torch.float16)
    top_idx = torch.zeros((T, K), device=dev, dtype=torch.long)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        block = idx[s:e]
        if stream:
            block = block.to(dev, non_blocking=True)
        sims = Qt @ block.T                                       # [T, chunk]
        cat_v = torch.cat([top_val, sims], 1)
        cat_i = torch.cat([top_idx, torch.arange(s, e, device=dev).expand(T, e - s)], 1)
        top_val, sel = cat_v.topk(K, dim=1)
        top_idx = torch.gather(cat_i, 1, sel)
        if stream:
            del block, sims, cat_v, cat_i
    return top_val, top_idx
