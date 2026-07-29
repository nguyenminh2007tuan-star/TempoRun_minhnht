"""Stage 3.5 — rerank top-k candidates with Qwen3-VL-Reranker (cross-encoder).

The bi-encoders (CLIP family / Qwen3-VL-Embedding) retrieve candidates; this model
re-scores each (full query, keyframe image) PAIR with cross-attention, which is what
finally fixes the ordering — per the Qwen3-VL paper, Reranker-8B lifts MMEB-v2 video
retrieval 53.6 -> 61.0 over embedding-only, while Reranker-2B does NOT help video
(53.6 -> 53.2), so only the 8B variant is worth running here.

The reranker reads the FULL query (long ctx) — no decomposition needed at this stage.

VRAM: 9B params. fp16 ≈ 18GB -> must shard across two 11GB cards:
  CUDA_VISIBLE_DEVICES=4,6 python rerank.py ... (device_map=auto handles the split)
sm_75 (2080 Ti) has no bf16: we load fp16 and fall back to fp32-sharded only if
scores come back NaN.

Input candidates JSON (from retrieve_fusion.py --candidates-out):
  {"T0001": [{"video_id": "...", "frame_ms": 123, "score": 0.9}, ...], ...}
Output: standard submission.json (same schema as retrieve.py).
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np

from retrieval_core import top_frames

RERANK_PROMPT = "Retrieve images or text relevant to the user's query."


def kf_path(kf_root: str, video_id: str, frame_ms: int, _cache={}):
    """(video_id, frame_ms) -> keyframe jpg path via the aligned ts_ms.npy index."""
    if video_id not in _cache:
        vdir = os.path.join(kf_root, video_id)
        ts = np.load(os.path.join(vdir, "ts_ms.npy"))
        files = sorted(glob.glob(os.path.join(vdir, "k_*.jpg")))
        _cache[video_id] = (ts[: len(files)], files)
    ts, files = _cache[video_id]
    hit = np.nonzero(ts == frame_ms)[0]
    if len(hit) == 0:  # tolerate tiny drift: nearest frame
        hit = [int(np.abs(ts.astype(np.int64) - frame_ms).argmin())]
    return files[int(hit[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="candidates json from retrieve_fusion")
    ap.add_argument("--tasks", required=True, help="round task jsonl (for full queries)")
    ap.add_argument("--keyframes", required=True, help="keyframes root (local disk)")
    ap.add_argument("--out", required=True, help="reranked submission.json")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-Reranker-8B")
    ap.add_argument("--rerank-depth", type=int, default=20,
                    help="how many candidates per task to rescore")
    ap.add_argument("--top-k", type=int, default=10, help="final predictions per task")
    ap.add_argument("--min-gap-ms", type=int, default=0,
                    help="temporal NMS within a video (0 = pure rerank-score order)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--scores-out", default=None,
                    help="lưu điểm thô rerank+fusion từng candidate để quét cách trộn offline")
    ap.add_argument("--int8", action="store_true",
                    help="load in 8-bit (bitsandbytes) — weights ~8.5GB, leaves little for the "
                         "~2.3GB single-image activation on an 11GB card")
    ap.add_argument("--nf4", action="store_true",
                    help="load in 4-bit NF4 — weights ~5.5GB, comfortably fits activations on 11GB")
    ap.add_argument("--limit-tasks", type=int, default=0, help="smoke test: only first N tasks")
    args = ap.parse_args()

    cands = json.load(open(args.candidates))
    queries = {t["task_id"]: t["description"] for t in
               (json.loads(l) for l in open(args.tasks))}
    print(f"[rerank] {len(cands)} tasks, depth={args.rerank_depth}", flush=True)

    import torch
    from sentence_transformers import CrossEncoder
    mk = {"dtype": torch.float16}
    if args.nf4:
        from transformers import BitsAndBytesConfig
        mk["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        mk["device_map"] = {"": 0}
        mode = "nf4"
    elif args.int8:
        from transformers import BitsAndBytesConfig
        mk["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        mk["device_map"] = {"": 0}
        mode = "int8"
    else:
        mk["device_map"] = "auto"
        mode = "fp16"
    model = CrossEncoder(args.model, model_kwargs=mk)
    print(f"[rerank] {args.model} loaded ({mode})", flush=True)

    items = sorted(cands.items())
    if args.limit_tasks:
        items = items[:args.limit_tasks]
    preds = []
    all_scores = {}
    for ti, (task_id, cand_list) in enumerate(items):
        cand_list = cand_list[: args.rerank_depth]
        query = queries[task_id]
        pairs = [(query, {"image": kf_path(args.keyframes, c["video_id"], c["frame_ms"])})
                 for c in cand_list]
        scores = model.predict(pairs, batch_size=args.batch_size, prompt=RERANK_PROMPT)
        scores = np.asarray(scores, dtype=np.float64)
        if np.isnan(scores).any():
            raise SystemExit("NaN scores — fp16 overflow on sm_75; rerun with fp32 or int8")

        # Rank by rerank score alone; each frame keeps its own video_id (no dedup —
        # the metric rewards the first in-interval hit, so a strong video's 2nd frame
        # should outrank a weak video's 1st). top_frames handles optional temporal NMS.
        scored = [(c["video_id"], int(c["frame_ms"]), float(sc))
                  for c, sc in zip(cand_list, scores)]
        if args.scores_out is not None:
            # Lưu điểm thô của CẢ HAI tầng để sau này quét cách trộn trên CPU,
            # khỏi phải chạy lại GPU cho từng hệ số alpha.
            all_scores[task_id] = [
                {"video_id": c["video_id"], "frame_ms": int(c["frame_ms"]),
                 "rerank": float(sc), "fusion": float(c.get("score", 0.0))}
                for c, sc in zip(cand_list, scores)]
        picked = top_frames(scored, top_k=args.top_k, min_gap_ms=args.min_gap_ms)
        results = [{"rank": rank, "video_id": v, "frame_ms": fms}
                   for rank, (v, fms, sc) in enumerate(picked, 1)]
        preds.append({"task_id": task_id, "results": results})
        if (ti + 1) % 25 == 0:
            print(f"[rerank] {ti+1}/{len(cands)}", flush=True)

    if args.scores_out is not None:
        json.dump(all_scores, open(args.scores_out, "w"))
        print(f"[rerank] điểm thô -> {args.scores_out}", flush=True)
    json.dump({"predictions": preds}, open(args.out, "w"))
    print(f"[rerank] DONE -> {args.out} ({len(preds)} tasks)", flush=True)


if __name__ == "__main__":
    main()
