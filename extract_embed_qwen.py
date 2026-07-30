"""Stage 2 variant — embed keyframes with Qwen3-VL-Embedding (a VLM bi-encoder).

Kept separate from extract_embed.py because this model is not an open_clip encoder:
it loads through sentence-transformers and takes {"image": path} documents rather
than preprocessed tensors, so it shares no code with the ClipModel path. What it
does share is the on-disk contract — `<out>/shards/<video_id>.npz` with emb[K,D]
fp16 + ts_ms[K] int32 — so retrieve_fusion.py can treat its index like any other.

Expect this to be slow: unlike CLIP's single vision tower, every frame runs a full
LLM forward, and on sm_75 there is no FlashAttention. It is resumable per video.
"""
from __future__ import annotations
import argparse, glob, os, time
from pathlib import Path
import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def list_keyframe_dirs(kf_root):
    return [Path(p).parent for p in sorted(glob.glob(os.path.join(kf_root, "*", "ts_ms.npy")))]


def list_frames(vdir):
    files = sorted(glob.glob(os.path.join(vdir, "k_*.jpg")))
    ts = np.load(os.path.join(vdir, "ts_ms.npy"))
    n = min(len(files), len(ts))
    return files[:n], [int(t) for t in ts[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    # fp16 by default because it runs on every card this was tested on, T4 (sm_75)
    # included. The shipped index_qwen8b was produced with bf16 on the teammate's
    # machine; bf16 has no hardware support before Ampere, so it is opt-in here.
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    shard_dir = Path(args.out) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    fail_log = Path(args.out) / f"failed_embed_shard{args.shard_index}.txt"

    vdirs = list_keyframe_dirs(args.keyframes)
    if not vdirs:
        raise SystemExit(f"no keyframes in {args.keyframes}")
    mine = [d for i, d in enumerate(vdirs) if i % args.shard_count == args.shard_index]
    if args.limit:
        mine = mine[:args.limit]
    print(f"[shard {args.shard_index}/{args.shard_count}] {len(mine)}/{len(vdirs)} videos", flush=True)

    import torch
    from sentence_transformers import SentenceTransformer
    dtype = getattr(torch, args.dtype)
    model = SentenceTransformer(args.model, device=args.device,
                                model_kwargs={"dtype": dtype})
    print(f"[qwen] {args.model} on {args.device} {args.dtype}", flush=True)

    t0 = time.time(); done = nframes = failed = 0
    for vdir in mine:
        vid = vdir.name
        out_npz = shard_dir / f"{vid}.npz"
        if out_npz.exists():
            done += 1
            continue
        try:
            files, ts = list_frames(vdir)
            if not files:
                raise RuntimeError("no frames")
            emb = model.encode([{"image": f} for f in files],
                               batch_size=args.batch_size,
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)
            np.savez(out_npz, emb=emb.astype(np.float16), ts_ms=np.asarray(ts, dtype=np.int32))
            nframes += len(ts)
        except Exception as ex:
            failed += 1
            with open(fail_log, "a") as f:
                f.write(f"{vid}\t{ex}\n")
        done += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f"[shard {args.shard_index}] {done}/{len(mine)} | {nframes} frames | "
                  f"fail={failed} | {done/el*60:.1f} vids/min | "
                  f"ETA {(len(mine)-done)/max(done/el,1e-9)/3600:.1f}h", flush=True)
    print(f"[shard {args.shard_index}] DONE {done} videos, {nframes} frames, {failed} failed, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
