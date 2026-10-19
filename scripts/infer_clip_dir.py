#!/usr/bin/env python3
"""Run Any4D over a dumped clip directory and write depth in the U4DR cache layout.

WHY THIS LIVES HERE, in the Any4D repository rather than in unified4DReconstructor:
everything below is a fact about Any4D -- which checkpoint, which hydra overrides, what
resolution it was trained at, which of its outputs is metric depth -- so it belongs with
the model, versioned against the model's own history. U4DR keeps only the thin replay
adapter (``eval/cached_depth.py``), which needs to know nothing about Any4D.

WHY OUT OF PROCESS. Any4D pins ``torch~=2.6.0``; U4DR runs torch 2.12 on cu130, and two
of Any4D's dependencies are git-pinned. Installing either side into the other breaks it.
MoGe-3 already runs this way in that project, for the same class of reason.

INPUT is what ``scripts/eval/dump_eval_clips.py`` writes, so every method is scored on
byte-identical frames in the same order::

    <clips>/clips/<clip_idx:06d>/<frame_idx:02d>.png
    <clips>/manifest.json

OUTPUT is the layout ``eval/cached_depth.py`` replays. The rich form is used, because
Any4D emits far more than depth and scoring it on depth alone would understate it::

    <out>/<clip_idx:06d>/<frame_idx:02d>.npz
        depth       (H, W)      float32, 0 = invalid   -> depth metrics
        ray         (6, H, W)   world origin + unit dir -> point-cloud F-score, Chamfer
        extrinsics  (4, 4)      world-to-camera         -> pose AUC

Three of U4DR's five scored axes. The two it cannot supply are segmentation (Any4D has
no semantic head) and 2-D optical flow (it emits 3-D scene flow, which is a different
quantity -- projecting one to the other is a modelling choice, not a format conversion,
so it is deliberately left out rather than fudged).

TWO THINGS THAT WOULD SILENTLY CORRUPT THE COMPARISON, handled explicitly:

* ``depth_along_ray`` is distance along the viewing ray, not the z component. U4DR's
  depth is also along-ray (``unproject`` is ``o + depth * d`` with unit ``d``), so the
  two agree and nothing is converted. If Any4D ever changes that, fix it HERE.
* Any4D infers at 518x336 and U4DR scores on the ground-truth grid. The depth is resized
  back to each clip's own frame size before saving, in LOG space -- resampling depth
  linearly across an occlusion edge invents surfaces in the gap, which is the exact
  artefact U4DR's boundary metric measures.

    uv run python scripts/infer_clip_dir.py \
        --clips /home/mas/data/eval_clips/test \
        --out   /home/mas/data/eval_preds/any4d \
        --checkpoint_path checkpoints/any4d_4v_combined.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RESOLUTION = (518, 336)          # what the released checkpoint was trained at


def _quat_to_rot(q: "torch.Tensor") -> "torch.Tensor":
    """Unit quaternion (w, x, y, z) -> 3x3 rotation. Any4D stores camera-to-world."""
    q = q / q.norm().clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="dir holding clips/ and manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint_path", default="checkpoints/any4d_4v_combined.pth")
    ap.add_argument("--config_dir", default="configs")
    ap.add_argument("--machine", default="local")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N clips")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from any4d.utils.image import load_images
    from any4d.utils.moge_inference import load_moge_model
    from demo_inference import init_inference_model, sample_inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    high_level_config = {
        "path": f"{args.config_dir}/train.yaml",
        "config_overrides": [
            f"machine={args.machine}",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            "model/task=images_only",
        ],
        "checkpoint_path": args.checkpoint_path,
        "trained_with_amp": True,
        "data_norm_type": "dinov2",
    }
    model = init_inference_model(high_level_config, args.checkpoint_path, device)
    moge_model = load_moge_model(device=str(device))

    clips_root = Path(args.clips) / "clips"
    manifest = json.loads((Path(args.clips) / "manifest.json").read_text())
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    clip_dirs = sorted(d for d in clips_root.iterdir() if d.is_dir())
    if args.limit:
        clip_dirs = clip_dirs[: args.limit]

    written = 0
    for ci, cdir in enumerate(clip_dirs):
        frames = sorted(cdir.glob("*.png"))
        if not frames:
            continue
        gt_h, gt_w = Image.open(frames[0]).size[::-1]      # the grid U4DR scores on
        views = load_images(
            [str(f) for f in frames],
            size=RESOLUTION,
            verbose=False,
            norm_type=high_level_config["data_norm_type"],
            patch_size=14,
            compute_moge_mask=True,
            moge_model=moge_model,
            binary_mask_path=None,
        )
        with torch.no_grad():
            pred = sample_inference(model, views, device,
                                    use_amp=high_level_config["trained_with_amp"])
        dst = out_root / cdir.name
        dst.mkdir(parents=True, exist_ok=True)
        for fi in range(len(frames)):
            d = pred[f"pred{fi + 1}"]["depth_along_ray"].squeeze().float().cpu()
            if d.ndim == 3:                                 # (H,W,1) or (1,H,W)
                d = d[0] if d.shape[0] == 1 else d[..., 0]
            if tuple(d.shape) != (gt_h, gt_w):
                # log space: a linear resample across an occlusion edge lands in the
                # empty gap between two surfaces and invents depth that exists nowhere.
                lg = torch.log(d.clamp_min(1e-6))[None, None]
                d = torch.exp(F.interpolate(lg, size=(gt_h, gt_w),
                                            mode="bilinear", align_corners=False))[0, 0]
            entry = {"depth": d.numpy().astype(np.float32)}

            pv = pred[f"pred{fi + 1}"]
            # WORLD rays. Any4D gives per-pixel directions in the CAMERA frame plus the
            # camera pose; U4DR's ray is (origin, unit direction) in the WORLD frame, so
            # the direction is rotated by R^T and the origin is the camera centre. Doing
            # this here rather than in U4DR keeps the frame convention with the model
            # that defines it.
            ext = None
            if "cam_quats" in pv and "cam_trans" in pv:
                q = pv["cam_quats"].squeeze().float().cpu()
                t = pv["cam_trans"].squeeze().float().cpu()
                R = _quat_to_rot(q)                       # camera-to-world rotation
                ext = torch.eye(4)
                ext[:3, :3] = R.T                         # world-to-camera
                ext[:3, 3] = -R.T @ t
                entry["extrinsics"] = ext.numpy().astype(np.float32)

            if "ray_directions" in pv and ext is not None:
                rd = pv["ray_directions"].squeeze().float().cpu()
                if rd.shape[0] != 3:                      # (H, W, 3) -> (3, H, W)
                    rd = rd.permute(2, 0, 1)
                if rd.shape[-2:] != (gt_h, gt_w):
                    rd = F.interpolate(rd[None], size=(gt_h, gt_w), mode="nearest")[0]
                rd = rd / rd.norm(dim=0, keepdim=True).clamp_min(1e-8)
                world_d = (R @ rd.reshape(3, -1)).reshape(3, gt_h, gt_w)
                origin = t.reshape(3, 1, 1).expand(3, gt_h, gt_w)
                entry["ray"] = torch.cat([origin, world_d], 0).numpy().astype(np.float32)

            np.savez(dst / f"{fi:02d}.npz", **entry)
            written += 1
        if ci % 20 == 0:
            print(f"[any4d] {ci + 1}/{len(clip_dirs)} clips", flush=True)

    print(f"[any4d] wrote {written} depth maps for {len(clip_dirs)} clips -> {out_root}")
    print(f"[any4d] the dump's manifest lists {len(manifest.get('clips', []))} clips; a "
          f"mismatch means the comparison would score missing clips as invalid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
