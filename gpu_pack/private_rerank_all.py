"""Stage 6 — rerank the fusion pool with Qwen3-VL-Reranker-8B (fp16), four passes, one model load.

The four passes are the four signals the post-process consumes:
  scores_1f_raw    single keyframe x ORIGINAL query      -> ranking (the anchor submission)
  scores_1f_strip  single keyframe x scaffolding-stripped query
  scores_clip3     3-frame strip  x stripped query, top-20  -> centroid weight field
  scores_clip5     5-frame strip  x stripped query, top-20  -> centroid weight field / consensus

Passes 3 and 4 are not a second opinion on ranking: their score field is what the
rank-1 centroid nudge integrates over, which is where most of the gain came from.
Batch sizes shrink with panel width because a 5-panel strip is ~5x the vision tokens.

Example
-------
python private_rerank_all.py \
    --images rerank_inputs \
    --pool   outputs/candidates_SQ8_private_d200.json \
    --tasks  data/private_round_tasks.jsonl \
    --strip  precomputed/private_queries_stripped.json \
    --manifest rerank_inputs/clip_top20_private.json \
    --out-dir outputs

Needs a CUDA GPU with >= 20 GB free (fp16 8B weights are ~16 GB). If a pass dies with
an OOM or prints NaN, lower that pass's batch size (--batch-1f / --batch-c3 / --batch-c5).
Each pass writes its file as soon as it finishes, so a later failure never loses earlier work.
"""
import argparse, glob, json, os, time
import numpy as np
import torch

REPO = "Qwen/Qwen3-VL-Reranker-8B"
# Pinned revision: the reranker's scores define the submitted ranking, so the exact
# weights matter for reproduction.
LAB  = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"

PROMPT_1F = "Retrieve images or text relevant to the user's query."
PROMPT_C3 = ("Retrieve images or text relevant to the user's query. Each image shows three "
             "consecutive moments (2 seconds apart) of one video, left to right.")
PROMPT_C5 = ("Retrieve images or text relevant to the user's query. Each image shows five "
             "consecutive moments of one video, 4 seconds apart, left to right. The MIDDLE "
             "panel is the moment in question; the others are its before/after context.")

def load_model():
    assert torch.cuda.is_available(), "CUDA GPU required"
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(REPO, revision=LAB, max_workers=8)
    # Some releases of this repo ship a modules.json pointing at a directory name that
    # does not exist in the snapshot; repair it so sentence-transformers can load.
    mp = os.path.join(local_dir, "modules.json")
    mods = json.load(open(mp)); fixed = False
    for m in mods:
        p = m.get("path", "")
        if p and not os.path.isdir(os.path.join(local_dir, p)):
            real = [d for d in os.listdir(local_dir)
                    if os.path.isdir(os.path.join(local_dir, d)) and d.endswith("LogitScore")]
            if real:
                m["path"] = real[0]; fixed = True
    if fixed:
        os.remove(mp); json.dump(mods, open(mp, "w"), indent=2)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(local_dir, model_kwargs={"dtype": torch.float16, "device_map": {"": 0}})
    print("[model] fp16, single card", flush=True)
    return model

def index_dir(root, name):
    return {os.path.basename(p): p for p in glob.glob(os.path.join(root, name, "*.jpg"))}

def run_1f(model, images, pool, qmap, depth, out_path, batch):
    IMG = index_dir(images, "frames_1f")
    assert IMG, f"no frames in {images}/frames_1f — run build_pod_private.py first"
    cands = json.load(open(pool))
    from tqdm.auto import tqdm
    res, t0 = {}, time.time()
    for tid, cl in tqdm(sorted(cands.items()), desc=os.path.basename(out_path), unit="task"):
        cl = [c for c in cl[:depth] if f"{c['video_id']}__{int(c['frame_ms'])}.jpg" in IMG]
        if not cl:
            continue
        pairs = [(qmap[tid], {"image": IMG[f"{c['video_id']}__{int(c['frame_ms'])}.jpg"]}) for c in cl]
        sc = np.asarray(model.predict(pairs, batch_size=batch, prompt=PROMPT_1F), dtype=np.float64)
        if np.isnan(sc).any():
            raise SystemExit("NaN scores — lower the batch size for this pass")
        res[tid] = [{"video_id": c["video_id"], "frame_ms": int(c["frame_ms"]),
                     "rerank": float(s), "fusion": float(c.get("score", 0.0))}
                    for c, s in zip(cl, sc)]
    json.dump(res, open(out_path, "w"))
    print(f"  -> {out_path} ({(time.time()-t0)/60:.1f} min)", flush=True)

def run_clip(model, images, subdir, manifest, qmap, prompt, out_path, batch):
    IMG = index_dir(images, subdir)
    assert IMG, f"no strips in {images}/{subdir} — run build_pod_private.py first"
    man = json.load(open(manifest))
    from tqdm.auto import tqdm
    res, t0 = {}, time.time()
    for tid, cl in tqdm(sorted(man.items()), desc=os.path.basename(out_path), unit="task"):
        ok = [(v, ms) for v, ms in cl if f"{v}__{ms}.jpg" in IMG]
        if not ok:
            continue
        pairs = [(qmap[tid], {"image": IMG[f"{v}__{ms}.jpg"]}) for v, ms in ok]
        sc = np.asarray(model.predict(pairs, batch_size=batch, prompt=prompt), dtype=np.float64)
        if np.isnan(sc).any():
            raise SystemExit("NaN scores — lower the batch size for this pass")
        res[tid] = [{"video_id": v, "frame_ms": int(ms), "clip_score": float(s)}
                    for (v, ms), s in zip(ok, sc)]
    json.dump(res, open(out_path, "w"))
    print(f"  -> {out_path} ({(time.time()-t0)/60:.1f} min)", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir from build_pod_private.py (frames_1f/, clip3/, clip5/)")
    ap.add_argument("--pool", required=True, help="candidate pool json from retrieve_fusion.py")
    ap.add_argument("--tasks", required=True, help="round task file (.jsonl) — the ORIGINAL queries")
    ap.add_argument("--strip", required=True, help="stripped queries json from strip_queries.py")
    ap.add_argument("--manifest", required=True, help="clip_top20_*.json from build_pod_private.py")
    ap.add_argument("--out-dir", default=".", help="where the four score files are written")
    # Depth 100 is the setting the submitted results were produced with; the pool file is
    # deeper only so a d200 variant stays possible without re-running fusion.
    ap.add_argument("--depth", type=int, default=100, help="pool depth to rerank (default 100 = submitted setting)")
    ap.add_argument("--batch-1f", type=int, default=8)
    ap.add_argument("--batch-c3", type=int, default=4)
    ap.add_argument("--batch-c5", type=int, default=2)
    args = ap.parse_args()

    q_raw = {t["task_id"]: t["description"] for t in (json.loads(l) for l in open(args.tasks))}
    q_str = json.load(open(args.strip))
    assert set(q_raw) == set(q_str), "task ids differ between --tasks and --strip"
    os.makedirs(args.out_dir, exist_ok=True)
    o = lambda fn: os.path.join(args.out_dir, fn)

    model = load_model()
    run_1f(model, args.images, args.pool, q_raw, args.depth, o("scores_1f_raw.json"), args.batch_1f)
    run_1f(model, args.images, args.pool, q_str, args.depth, o("scores_1f_strip.json"), args.batch_1f)
    run_clip(model, args.images, "clip3", args.manifest, q_str, PROMPT_C3, o("scores_clip3.json"), args.batch_c3)
    run_clip(model, args.images, "clip5", args.manifest, q_str, PROMPT_C5, o("scores_clip5.json"), args.batch_c5)
    print(f"\nAll four passes done -> {args.out_dir}", flush=True)

if __name__ == "__main__":
    main()
