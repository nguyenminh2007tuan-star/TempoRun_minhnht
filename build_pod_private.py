"""Stage 5 — build the reranker's image inputs from the fusion candidate pool.

Produces three things the reranker consumes:
  frames_1f/  single keyframe per pooled candidate  (depth-100 rerank passes)
  clip3/      3-frame horizontal strip, top-20      [t-2s, t, t+2s]
  clip5/      5-frame horizontal strip, top-20      [t-8s, t-4s, t, t+4s, t+8s]
  clip_top20_<round>.json  manifest of the top-20 candidates per task

The strips are why the centroid step works: a single frame cannot tell the reranker
where a shot begins or ends, but a strip of neighbouring moments can, and its score
field is what the rank-1 nudge integrates over. MIDDLE panel is always the candidate
moment itself.

Resumable: files that already exist are skipped, so a re-run continues where it stopped.

Example
-------
python build_pod_private.py \
    --keyframes keyframes \
    --pool outputs/candidates_SQ8_private_d200.json \
    --out rerank_inputs
"""
import argparse, json, os, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

CLIP3 = [-2000, 0, 2000]
CLIP5 = [-8000, -4000, 0, 4000, 8000]

# Set once in main() and inherited by the worker processes (fork on Linux).
KF = OUT = None

_cache = {}
def _load(v):
    if v not in _cache:
        vdir = os.path.join(KF, v)
        ts = np.load(os.path.join(vdir, "ts_ms.npy"))
        files = sorted(glob.glob(os.path.join(vdir, "k_*.jpg")))
        _cache[v] = (ts[:len(files)].astype(np.int64), files)
    return _cache[v]

def path_near(v, target):
    """Nearest extracted keyframe to `target` ms. Tolerates a keyframe set that differs
    slightly from the one the shipped index was built on (different ffmpeg build)."""
    ts, files = _load(v)
    return files[int(np.abs(ts - target).argmin())]

def copy_one(args):
    import shutil
    v, ms = args
    dst = f"{OUT}/frames_1f/{v}__{ms}.jpg"
    if os.path.exists(dst):
        return 0
    try:
        shutil.copyfile(path_near(v, ms), dst)
        return 1
    except Exception as e:
        print(f"COPY FAIL {v}@{ms}: {e}", flush=True)
        return -1

def make_strip(paths, out):
    ims = [Image.open(p).convert("RGB") for p in paths]
    h = max(im.height for im in ims)
    ims = [im if im.height == h else im.resize((max(1, round(im.width * h / im.height)), h)) for im in ims]
    canvas = Image.new("RGB", (sum(im.width for im in ims), h))
    x = 0
    for im in ims:
        canvas.paste(im, (x, 0)); x += im.width
    canvas.save(out, quality=90)

def strip_one(args):
    v, ms = args
    o3 = f"{OUT}/clip3/{v}__{ms}.jpg"
    o5 = f"{OUT}/clip5/{v}__{ms}.jpg"
    try:
        if not os.path.exists(o3):
            make_strip([path_near(v, ms + d) for d in CLIP3], o3)
        if not os.path.exists(o5):
            make_strip([path_near(v, ms + d) for d in CLIP5], o5)
        return 1
    except Exception as e:
        print(f"STRIP FAIL {v}@{ms}: {e}", flush=True)
        return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True,
                    help="keyframe root from extract_keyframes.py: <root>/<video_id>/{k_*.jpg, ts_ms.npy}")
    ap.add_argument("--pool", required=True, help="candidate pool json from retrieve_fusion.py --candidates-out")
    ap.add_argument("--out", required=True, help="output dir; gets frames_1f/, clip3/, clip5/ and the manifest")
    ap.add_argument("--manifest-name", default="clip_top20_private.json")
    ap.add_argument("--depth", type=int, default=100,
                    help="how deep into the pool to prepare single frames (must be >= the reranker's DEPTH)")
    ap.add_argument("--top-strips", type=int, default=20, help="how many top candidates get 3f/5f strips")
    ap.add_argument("--nproc", type=int, default=min(32, (os.cpu_count() or 8)))
    args = ap.parse_args()

    global KF, OUT
    KF, OUT = args.keyframes, args.out
    for d in ("frames_1f", "clip3", "clip5"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)

    pool = json.load(open(args.pool))
    f1, top, manifest = set(), set(), {}
    for tid, lst in pool.items():
        for c in lst[:args.depth]:
            f1.add((c["video_id"], int(c["frame_ms"])))
        head = [(c["video_id"], int(c["frame_ms"])) for c in lst[:args.top_strips]]
        manifest[tid] = [[v, ms] for v, ms in head]
        top.update(head)
    json.dump(manifest, open(os.path.join(OUT, args.manifest_name), "w"))
    print(f"[manifest] {args.manifest_name} ({len(manifest)} tasks)", flush=True)
    print(f"[plan] frames_1f {len(f1)} | strips {len(top)} x2", flush=True)

    print("[1/2] copying single frames ...", flush=True)
    with ProcessPoolExecutor(args.nproc) as ex:
        for i, _ in enumerate(ex.map(copy_one, f1, chunksize=64), 1):
            if i % 20000 == 0:
                print(f"   {i}/{len(f1)}", flush=True)

    print("[2/2] building clip3 + clip5 strips ...", flush=True)
    with ProcessPoolExecutor(args.nproc) as ex:
        for i, _ in enumerate(ex.map(strip_one, top, chunksize=32), 1):
            if i % 4000 == 0:
                print(f"   {i}/{len(top)}", flush=True)

    n1 = len(glob.glob(f"{OUT}/frames_1f/*.jpg"))
    n3 = len(glob.glob(f"{OUT}/clip3/*.jpg"))
    n5 = len(glob.glob(f"{OUT}/clip5/*.jpg"))
    print(f"[done] {OUT}: frames_1f {n1} | clip3 {n3} | clip5 {n5}", flush=True)

if __name__ == "__main__":
    main()
