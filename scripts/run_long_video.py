#!/usr/bin/env python
"""Run Any4D over an arbitrarily long video in consecutive clips, streaming to disk.

Same clip layout and npz format as unified4DReconstructor/scripts/eval/run_long_video.py,
so the methods' per-frame outputs line up frame for frame. Model construction, input
size, normalisation and AMP follow scripts/demo_inference.py. Each clip goes to
``<out>/clip_XXXXX.npz`` when it finishes; a clip already on disk is skipped (resume).

    uv run python scripts/run_long_video.py --frames <dir of jpg> --window 8 --out <dir>

Each npz: frames (n,) int32; depth (n,H,W) f16 (camera-frame z); xyz (n,3,H,W) f16
camera-frame points; scene_flow (n,3,H,W) f16;
conf (n,H,W) f16; cam_trans (n,3) / cam_quats (n,4) f32, relative to the clip's first view.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from demo_inference import init_inference_model
from probe_frame_scaling import CONFIG

from any4d.utils.image import load_images
from any4d.utils.inference import loss_of_one_batch_multi_view


def _stack(preds, key, f):
    """Per-view prediction ``key`` (batch 0) through ``f``, stacked over the clip's views."""
    return np.stack([f(p[key][0].float()).cpu().numpy() for p in preds])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint_path", default="checkpoints/any4d_4v_combined.pth")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [str(p) for p in sorted(Path(args.frames).glob("*.jpg"))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [list(range(i, min(i + args.window, len(paths)))) for i in range(0, len(paths), args.window)]
    device = torch.device("cuda")
    model = init_inference_model(CONFIG, args.checkpoint_path, device)
    hw = None
    t0, done = time.time(), 0
    for ci, idx in enumerate(clips):
        dest = out / f"clip_{ci:05d}.npz"
        if dest.exists():
            continue
        views = load_images([paths[i] for i in idx], size=(518, 336), norm_type="dinov2", patch_size=14)
        with torch.no_grad():
            r = loss_of_one_batch_multi_view(views, model, None, device, use_amp=True)
        preds = [r[f"pred{k + 1}"] for k in range(len(idx))]
        if hw is None:
            hw = list(preds[0]["pts3d_cam"].shape[1:3])
            (out / "source.json").write_text(json.dumps(
                {"method": "Any4D", "ckpt": str(Path(args.checkpoint_path).resolve()),
                 "frames": args.frames, "n_frames": len(paths), "window": args.window,
                 "input_hw": hw}, indent=2))
            print(f"[long] Any4D: {len(paths)} frames, {len(clips)} clips of {args.window} "
                  f"@ {hw[0]}x{hw[1]} -> {out}", flush=True)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, frames=np.asarray(idx, np.int32),
            depth=_stack(preds, "pts3d_cam", lambda t: t[..., 2]).astype(np.float16),
            xyz=_stack(preds, "pts3d_cam", lambda t: t.permute(2, 0, 1)).astype(np.float16),
            scene_flow=_stack(preds, "scene_flow", lambda t: t.permute(2, 0, 1)).astype(np.float16),
            conf=_stack(preds, "conf", lambda t: t).astype(np.float16),
            cam_trans=_stack(preds, "cam_trans", lambda t: t),
            cam_quats=_stack(preds, "cam_quats", lambda t: t))
        tmp.rename(dest)
        done += 1
        if done % 25 == 0 or ci == len(clips) - 1:
            el = time.time() - t0
            left = sum(not (out / f"clip_{k:05d}.npz").exists() for k in range(ci + 1, len(clips)))
            print(f"[long] clip {ci + 1}/{len(clips)} elapsed {el / 60:.1f} min, "
                  f"ETA {el / done * left / 60:.1f} min", flush=True)
    print(f"[long] DONE {len(clips)} clips -> {out}", flush=True)


if __name__ == "__main__":
    main()
