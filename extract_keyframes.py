"""Stage 1 — extract I-frames (keyframes) per video via ffmpeg. NO model, NO embedding.

For every video we run ffmpeg with `-skip_frame nokey` (the decoder emits only
keyframes) plus the `showinfo` filter to recover each keyframe's `pts_time`
(-> clip-relative timestamp). The keyframe JPGs and their timestamps are written
to disk so that embedding (Stage 2, `extract_embed.py`) is a completely separate
step — you can re-embed the SAME frames with a different encoder without
re-decoding any video.

Output layout (one folder per video):
  <out>/<video_id>/k_00001.jpg, k_00002.jpg, ...   keyframe images (q:v 3)
  <out>/<video_id>/ts_ms.npy                        int32[K] timestamps, aligned to sorted jpgs

`ts_ms.npy` is written last and acts as the "done" marker:
  - Resumable: a video whose `ts_ms.npy` exists is skipped.
  - Shardable across processes: --shard-index / --shard-count.

Method ref: I-frame extraction with ffmpeg (`-skip_frame nokey`).
"""
from __future__ import annotations
import argparse, glob, os, re, shutil, subprocess, time
from pathlib import Path
import numpy as np

PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def _ffmpeg_bin():
    b = shutil.which("ffmpeg")
    if b:
        return b
    try:  # bundled static ffmpeg (cluster nodes lack a system ffmpeg)
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


FFMPEG = _ffmpeg_bin()


def _coll_from_path(path: str) -> str:
    up = path.upper()
    for c in ("V3C1", "V3C2"):
        if c in up:
            return c.lower()
    return "v3c"


def list_videos(roots):
    vids = []
    for root in roots:
        coll = Path(root).name.lower()  # V3C1 -> v3c1
        for mp4 in sorted(glob.glob(os.path.join(root, "videos", "*", "*.mp4"))):
            vids.append((f"{coll}_{Path(mp4).stem}", mp4))
    return vids


def list_from_file(path):
    vids = []
    for line in open(path):
        mp4 = line.strip()
        if mp4:
            vids.append((f"{_coll_from_path(mp4)}_{Path(mp4).stem}", mp4))
    return vids


def extract_keyframes(mp4: str, out_dir: str):
    """One ffmpeg pass. Writes k_*.jpg into out_dir; returns (jpg_paths, ts_ms) aligned."""
    os.makedirs(out_dir, exist_ok=True)
    pat = os.path.join(out_dir, "k_%05d.jpg")
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "info", "-skip_frame", "nokey",
           "-i", mp4, "-vsync", "0", "-vf", "showinfo", "-q:v", "3", pat]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    pts = [float(x) for x in PTS_RE.findall(proc.stderr.decode("utf-8", "ignore"))]
    files = sorted(glob.glob(os.path.join(out_dir, "k_*.jpg")))
    if not pts:  # fallback: spread evenly (rare — showinfo gave no pts)
        pts = list(range(len(files)))
    n = min(len(files), len(pts))
    for extra in files[n:]:  # drop any jpgs past known timestamps (rare misalignment)
        os.remove(extra)
    ts = [int(round(pts[i] * 1000)) for i in range(n)]
    return files[:n], ts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", action="append", default=[])
    p.add_argument("--video-list", default=None, help="file with one .mp4 path per line (overrides roots)")
    p.add_argument("--out", required=True, help="keyframes output dir (one subfolder per video)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="debug: cap #videos")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fail_log = out / f"failed_keyframes_shard{args.shard_index}.txt"

    vids = list_from_file(args.video_list) if args.video_list else list_videos(args.dataset_root)
    if not vids:
        raise SystemExit("no videos (need --dataset-root or --video-list)")
    mine = [v for i, v in enumerate(vids) if i % args.shard_count == args.shard_index]
    if args.limit:
        mine = mine[:args.limit]
    print(f"[shard {args.shard_index}/{args.shard_count}] {len(mine)}/{len(vids)} videos", flush=True)

    t0 = time.time(); done = nframes = failed = 0
    for vid, mp4 in mine:
        vdir = out / vid
        ts_path = vdir / "ts_ms.npy"
        if ts_path.exists():  # resumable
            done += 1
            continue
        try:
            files, ts = extract_keyframes(mp4, str(vdir))
            if not files:
                raise RuntimeError("no keyframes")
            np.save(ts_path, np.asarray(ts, dtype=np.int32))  # written last = done marker
            nframes += len(files)
        except Exception as ex:
            failed += 1
            shutil.rmtree(vdir, ignore_errors=True)  # don't leave a half-extracted folder
            with open(fail_log, "a") as f:
                f.write(f"{vid}\t{ex}\n")
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"[shard {args.shard_index}] {done}/{len(mine)} | {nframes} keyframes | "
                  f"fail={failed} | {done/el*60:.0f} vids/min | ETA {(len(mine)-done)/max(done/el,1e-9)/60:.0f}min",
                  flush=True)
    print(f"[shard {args.shard_index}] DONE {done} videos, {nframes} keyframes, {failed} failed, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
