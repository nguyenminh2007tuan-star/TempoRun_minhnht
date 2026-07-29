"""Stage 3 — text->keyframe retrieval -> submission.json.

Loads all per-video shards into one keyframe index, embeds each task description
with CLIP text encoder, finds the most similar keyframes (chunked top-k on GPU),
dedups to distinct videos, and emits up to 10 (video_id, frame_ms) predictions
per task — one frame (the matched keyframe's timestamp) per distinct video.
"""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
import numpy as np

from retrieval_core import load_index, chunked_topk, split_sentences


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True)
    p.add_argument("--tasks", required=True, help="a round's task file, e.g. public_round_tasks.jsonl")
    p.add_argument("--out", required=True, help="submission.json path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--top-videos", type=int, default=10)
    p.add_argument("--cand-keyframes", type=int, default=400)
    p.add_argument("--split-queries", action="store_true",
                    help="heuristic no-LLM decomposition: split each description into sentences before "
                         "encoding, fusing sub-query candidates back into their parent task via RRF (see "
                         "retrieve_fusion.py's docstring for why — long multi-shot queries silently truncate "
                         "at the encoder's context limit otherwise)")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF constant, only used with --split-queries")
    args = p.parse_args()

    import torch
    emb, vids, ts = load_index(args.shards)
    tasks = [json.loads(l) for l in open(args.tasks)]
    print(f"[tasks] {len(tasks)}", flush=True)
    T = len(tasks)

    from clip_model import ClipModel
    clip = ClipModel(args.model, args.pretrained, device=args.device, precision=args.precision)

    dev = args.device
    idx = torch.from_numpy(emb).to(dev).half()                    # [N, D]

    if not args.split_queries:
        Q = clip.encode_texts([t["description"] for t in tasks])  # [T, D] fp32
        Qt = torch.from_numpy(Q).to(dev).half()

        t0 = time.time()
        top_val, top_idx = chunked_topk(Qt, idx, args.cand_keyframes)
        print(f"[retrieve] scored {idx.shape[0]} keyframes x {Qt.shape[0]} queries in {time.time()-t0:.0f}s", flush=True)

        top_idx = top_idx.cpu().numpy()
        top_val = top_val.float().cpu().numpy()

        preds = []
        for ti, task in enumerate(tasks):
            seen = {}
            for r, sim in zip(top_idx[ti], top_val[ti]):
                v = str(vids[r])
                if v in seen:
                    continue
                seen[v] = (int(ts[r]), float(sim))
                if len(seen) >= args.top_videos:
                    break
            results = [{"rank": rank, "video_id": v, "frame_ms": center}
                       for rank, (v, (center, sim)) in enumerate(seen.items(), 1)]
            preds.append({"task_id": task["task_id"], "results": results})
    else:
        subq_text, subq_task = [], []
        for ti, t in enumerate(tasks):
            for s in split_sentences(t["description"]):
                subq_text.append(s)
                subq_task.append(ti)
        print(f"[split-queries] {T} tasks -> {len(subq_text)} sub-queries "
              f"({len(subq_text)/T:.1f} avg)", flush=True)

        Q = clip.encode_texts(subq_text)                          # [S, D] fp32
        Qt = torch.from_numpy(Q).to(dev).half()

        t0 = time.time()
        top_val, top_idx = chunked_topk(Qt, idx, args.cand_keyframes)
        print(f"[retrieve] scored {idx.shape[0]} keyframes x {Qt.shape[0]} sub-queries in {time.time()-t0:.0f}s", flush=True)
        top_idx = top_idx.cpu().numpy()

        fused = [defaultdict(float) for _ in range(T)]
        for si, ti in enumerate(subq_task):
            for rank, r in enumerate(top_idx[si], start=1):
                fused[ti][(str(vids[r]), int(ts[r]))] += 1.0 / (args.rrf_k + rank)

        preds = []
        for ti, task in enumerate(tasks):
            ranked = sorted(fused[ti].items(), key=lambda kv: kv[1], reverse=True)
            seen = {}
            for (v, center), score in ranked:
                if v in seen:
                    continue
                seen[v] = center
                if len(seen) >= args.top_videos:
                    break
            results = [{"rank": rank, "video_id": v, "frame_ms": center}
                       for rank, (v, center) in enumerate(seen.items(), 1)]
            preds.append({"task_id": task["task_id"], "results": results})

    sub = {"predictions": preds}
    json.dump(sub, open(args.out, "w"))
    print(f"[done] wrote {args.out} ({len(preds)} tasks)", flush=True)


if __name__ == "__main__":
    main()
