"""Stage 3 (multi-encoder) — text->keyframe retrieval fused across several
independently-embedded encoders -> submission.json.

Each encoder in --encoders (a JSON list of {name, shards, model, pretrained,
precision, device?}) was run through extract_embed.py against the SAME
keyframes from Stage 1, so every encoder's index shares the identical
(video_id, ts_ms) keyframe set. That lets us fuse purely by Reciprocal Rank
Fusion (RRF) keyed on (video_id, ts_ms) — no need to calibrate similarity
scales across encoders trained with different losses (e.g. SigLIP's sigmoid
loss vs. CLIP-style contrastive), which is the whole point of picking RRF over
a normalized weighted-sum of raw cosine scores.

  fused_score(v, ts) = sum over encoders where (v,ts) is in that encoder's
                        top-K candidates, of 1 / (rrf_k + rank_in_that_encoder)

Output schema is identical to retrieve.py's submission.json, so score.py needs
no changes.
"""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
import numpy as np

from retrieval_core import load_index, chunked_topk, split_sentences, top_frames


def load_encoders_config(path):
    cfgs = json.load(open(path))
    if not isinstance(cfgs, list) or not cfgs:
        raise SystemExit(f"{path} must be a non-empty JSON list of encoder configs")
    for c in cfgs:
        for req in ("name", "shards", "model", "pretrained"):
            if req not in c:
                raise SystemExit(f"encoder config missing required key {req!r}: {c}")
    return cfgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", required=True, help="JSON list of {name,shards,model,pretrained,precision,device}")
    p.add_argument("--tasks", required=True, help="a round's task file, e.g. public_round_tasks.jsonl")
    p.add_argument("--out", required=True, help="submission.json path")
    p.add_argument("--device", default="cuda:0", help="fallback device for encoders without their own 'device'")
    p.add_argument("--top-k", type=int, default=10, help="final predictions per task (metric allows ≤10)")
    p.add_argument("--min-gap-ms", type=int, default=0,
                    help="temporal NMS: drop a frame within this many ms of an already-picked "
                         "frame from the SAME video (0 = pure score ranking; ~4000 spreads picks "
                         "so dense near-duplicate keyframes don't waste prediction slots)")
    p.add_argument("--cand-keyframes", type=int, default=400,
                    help="candidate pool per encoder before fusion (larger than single-encoder retrieve.py's "
                         "default so encoders have enough overlap to actually reinforce each other)")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF constant (standard IR default)")
    p.add_argument("--index-root", default=".",
                    help="prefix for RELATIVE 'shards' paths in the encoders config; absolute "
                         "paths in the config are used as-is (default: current directory)")
    p.add_argument("--emb-root", default=".",
                    help="prefix for RELATIVE 'query_emb'/'subs_emb' paths in the encoders config; "
                         "absolute paths are used as-is (default: current directory)")
    p.add_argument("--cache", default=None,
                    help="per-encoder checkpoint path (default <out>.fused.pkl); makes fusion "
                         "resumable when a shared GPU evicts an encoder mid-run")
    p.add_argument("--chunk", type=int, default=50_000,
                    help="keyframes streamed to GPU per step; smaller = lower peak VRAM "
                         "(index stays on CPU, chunks copied on the fly)")
    p.add_argument("--decomposed", default=None,
                    help="JSON from decompose_queries.py ({task_id: {full, subs}}); LLM-written "
                         "self-contained sub-queries — preferred over --split-queries when available")
    p.add_argument("--subq-candidates-out", default=None,
                    help="dump per-sub-query candidate lists in narrated order, for order-aware "
                         "(temporal) rescoring downstream")
    p.add_argument("--candidates-out", default=None,
                    help="also dump the fused candidate pool per task ({task_id: [{video_id, frame_ms, "
                         "score}...]}) for rerank.py, before dedup-to-top-videos")
    p.add_argument("--split-queries", action="store_true",
                    help="heuristic no-LLM decomposition: split each description into sentences "
                         "before encoding (public-round queries average ~71 CLIP-BPE tokens, narrating "
                         "several shots per query — this avoids silent truncation at the encoder's context "
                         "limit; sub-queries are fused into their parent task via the same RRF mechanism "
                         "used across encoders)")
    args = p.parse_args()

    import torch
    from clip_model import ClipModel

    encoders = load_encoders_config(args.encoders)
    # Configs ship RELATIVE paths so the repo runs anywhere; --index-root / --emb-root
    # point them at wherever the downloaded index and precomputed embeddings actually live.
    # An absolute path in the config still wins, so an existing local setup keeps working.
    import os.path as _osp
    for enc in encoders:
        if enc.get("shards") and not _osp.isabs(enc["shards"]):
            enc["shards"] = _osp.join(args.index_root, enc["shards"])
        for key in ("query_emb", "subs_emb"):
            if enc.get(key) and not _osp.isabs(enc[key]):
                enc[key] = _osp.join(args.emb_root, enc[key])

    tasks = [json.loads(l) for l in open(args.tasks)]
    print(f"[tasks] {len(tasks)}", flush=True)
    T = len(tasks)

    # subq_pos[i] = position of sub-query i WITHIN its task, i.e. its slot in the
    # narrated order ("first scene", "then", "finally"). Kept so downstream can do
    # order-aware scoring; plain RRF ignores it.
    if args.decomposed:
        deco = json.load(open(args.decomposed))
        subq_text, subq_task, subq_pos = [], [], []
        for ti, t in enumerate(tasks):
            entry = deco.get(t["task_id"])
            subs = entry["subs"] if entry else split_sentences(t["description"])
            for j, s in enumerate(subs):
                subq_text.append(s)
                subq_task.append(ti)
                subq_pos.append(j)
        print(f"[decomposed] {T} tasks -> {len(subq_text)} LLM sub-queries "
              f"({len(subq_text)/T:.1f} avg)", flush=True)
    elif args.split_queries:
        subq_text, subq_task, subq_pos = [], [], []
        for ti, t in enumerate(tasks):
            for j, s in enumerate(split_sentences(t["description"])):
                subq_text.append(s)
                subq_task.append(ti)
                subq_pos.append(j)
        print(f"[split-queries] {T} tasks -> {len(subq_text)} sub-queries "
              f"({len(subq_text)/T:.1f} avg)", flush=True)
    else:
        subq_text = [t["description"] for t in tasks]
        subq_task = list(range(T))
        subq_pos = [0] * T

    # fused[task_idx][(video_id, ts_ms)] = accumulated RRF score.
    # Checkpointed after each encoder: on a shared GPU an encoder can be OOM-evicted
    # mid-run, and without this a crash on encoder 3 throws away encoders 1-2 (each a
    # ~10-30min search). Re-running skips encoders already folded in.
    import os, pickle
    cache_path = args.cache or (args.out + ".fused.pkl")
    done_names = set()
    fused = per_sub = None
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            ck = pickle.load(fh)
        if ck.get("subq_task") == subq_task and ck.get("n_tasks") == T and "per_sub" in ck:
            fused = [defaultdict(float, d) for d in ck["fused"]]
            per_sub = [[defaultdict(float, d) for d in task] for task in ck["per_sub"]]
            done_names = set(ck["done"])
            print(f"[resume] loaded checkpoint, {len(done_names)} encoders already fused: "
                  f"{sorted(done_names)}", flush=True)
        else:
            print("[resume] checkpoint lacks per-sub scores — starting fresh", flush=True)

    if fused is None:
        fused = [defaultdict(float) for _ in range(T)]
        nsub = [0] * T
        for ti, j in zip(subq_task, subq_pos):
            nsub[ti] = max(nsub[ti], j + 1)
        # per_sub[task][sub_position][(video,ts)] = RRF score from THAT sub-query alone,
        # so downstream can score a video by how well its frames match the sub-queries in
        # the narrated temporal order instead of as an order-free bag.
        per_sub = [[defaultdict(float) for _ in range(max(1, n))] for n in nsub]

    for enc in encoders:
        if enc["name"] in done_names:
            print(f"[encoder {enc['name']}] already in checkpoint, skip", flush=True)
            continue
        dev = enc.get("device", args.device)
        precision = enc.get("precision", "fp32")
        t0 = time.time()
        emb, vids, ts = load_index(enc["shards"])

        # Each encoder gets the query form it can actually consume. CLIP towers cap at
        # 64-77 tokens so they take the LLM-decomposed sub-queries; a long-context
        # encoder (Qwen3-VL-Embedding, 32k) takes the ORIGINAL query whole, which is
        # the one view no CLIP model in this ensemble can see.
        etype = enc.get("type", "clip")
        if enc.get("query", "subs") == "full":
            texts, text_task = [t["description"] for t in tasks], list(range(T))
        else:
            texts, text_task = subq_text, subq_task

        if enc.get("query_emb"):                                   # query vector tính sẵn (Qwen-8B encode trên Kaggle)
            d = np.load(enc["query_emb"], allow_pickle=True)
            qmap = {str(t): i for i, t in enumerate(d["task_ids"])}
            src = d["emb"].astype("float32")
            Q = np.stack([src[qmap[t["task_id"]]] for t in tasks])  # căn thứ tự task; yêu cầu query="full"
        elif enc.get("subs_emb"):                                  # SUBS tính sẵn (SigLIP encode trên Kaggle)
            d = np.load(enc["subs_emb"], allow_pickle=True)         # embed_siglip2_private.py
            smap = {(str(tid), int(p)): i for i, (tid, p)           # tra theo KEY (task_id,pos), không theo vị trí
                    in enumerate(zip(d["sub_task_id"], d["sub_pos"]))}
            src = d["emb"].astype("float32")
            # dựng Q theo ĐÚNG thứ tự subq_text/text_task fusion đang dùng
            Q = np.stack([src[smap[(tasks[ti]["task_id"], subq_pos[si])]]
                          for si, ti in enumerate(text_task)])
            print(f"[subs_emb] nạp {src.shape[0]} sub-vector tính sẵn, căn khớp {Q.shape[0]} subq", flush=True)
        elif etype == "qwen":
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(enc["model"], device=dev,
                                     model_kwargs={"dtype": torch.float16})
            Q = st.encode(texts, batch_size=8, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False).astype("float32")
            del st
        else:
            clip = ClipModel(enc["model"], enc["pretrained"], device=dev, precision=precision)
            Q = clip.encode_texts(texts)                           # [S, D] fp32, this encoder's own text tower
            del clip
        torch.cuda.empty_cache() if "cuda" in dev else None

        # Keep the index on CPU (pinned) and stream chunks to the GPU inside
        # chunked_topk: on a shared card the peak allocation is then ~one chunk,
        # not the whole ~1GB index, so a mid-run eviction by another user's job
        # is far less likely to OOM us.
        idx = torch.from_numpy(emb).half().pin_memory()
        Qt = torch.from_numpy(Q).to(dev).half()
        top_val, top_idx = chunked_topk(Qt, idx, args.cand_keyframes, chunk=args.chunk)
        top_idx = top_idx.cpu().numpy()
        del idx, Qt, top_val
        if "cuda" in dev:
            torch.cuda.empty_cache()

        uses_subs = enc.get("query", "subs") != "full"
        for si, ti in enumerate(text_task):  # si = query row for this encoder, ti = parent task
            pos = subq_pos[si] if uses_subs else None
            for rank, r in enumerate(top_idx[si], start=1):  # topk() is sorted desc -> rank 1 = best
                key = (str(vids[r]), int(ts[r]))
                w = enc.get("weight", 1.0) / (args.rrf_k + rank)   # weight: vd Qwen×8
                fused[ti][key] += w
                if pos is not None and pos < len(per_sub[ti]):
                    per_sub[ti][pos][key] += w

        done_names.add(enc["name"])
        with open(cache_path, "wb") as fh:
            pickle.dump({"done": sorted(done_names), "n_tasks": T, "subq_task": subq_task,
                         "fused": [dict(d) for d in fused],
                         "per_sub": [[dict(d) for d in task] for task in per_sub]}, fh)
        print(f"[encoder {enc['name']}] {emb.shape[0]} keyframes, {len(texts)} {enc.get('query','subs')}-queries, "
              f"{time.time()-t0:.0f}s | checkpointed", flush=True)

    if args.candidates_out:
        cand_dump = {}
        for ti, task in enumerate(tasks):
            ranked = sorted(fused[ti].items(), key=lambda kv: kv[1], reverse=True)
            cand_dump[task["task_id"]] = [
                {"video_id": v, "frame_ms": int(c), "score": s}
                for (v, c), s in ranked[:200]]  # d200 cho rerank
        json.dump(cand_dump, open(args.candidates_out, "w"))
        print(f"[candidates] wrote {args.candidates_out} (top-200/task, pre-dedup)", flush=True)

    if args.subq_candidates_out:
        # Per-sub-query candidates, kept in narrated order — the input for order-aware
        # (temporal) rescoring, which the collapsed RRF sum above cannot support.
        sub_dump = {}
        for ti, task in enumerate(tasks):
            sub_dump[task["task_id"]] = [
                [{"video_id": v, "frame_ms": int(c), "score": s}
                 for (v, c), s in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:100]]
                for d in per_sub[ti]]
        json.dump(sub_dump, open(args.subq_candidates_out, "w"))
        print(f"[subq-candidates] wrote {args.subq_candidates_out} (top-100 per sub-query)", flush=True)

    preds = []
    for ti, task in enumerate(tasks):
        scored = [(v, c, s) for (v, c), s in fused[ti].items()]
        picked = top_frames(scored, top_k=args.top_k, min_gap_ms=args.min_gap_ms)
        results = [{"rank": rank, "video_id": v, "frame_ms": int(c)}
                   for rank, (v, c, s) in enumerate(picked, 1)]
        preds.append({"task_id": task["task_id"], "results": results})

    sub = {"predictions": preds}
    json.dump(sub, open(args.out, "w"))
    print(f"[done] wrote {args.out} ({len(preds)} tasks, {len(encoders)} encoders fused)", flush=True)


if __name__ == "__main__":
    main()
