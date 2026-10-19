#!/usr/bin/env python
"""One multi-view inference pass over N consecutive frames; record time and peak GPU memory.

Model construction, input size, normalisation and AMP mirror scripts/demo_inference.py.
The MoGe mask the demo computes is skipped: it only refines the visualisation, and its
cost would be charged to Any4D. Same interface as the probes in unified4DReconstructor
and Open-d4rt, so the three can be compared run for run. An OOM is a result: it is
written to the record with status "oom".

    uv run python scripts/probe_frame_scaling.py --frames <dir of jpg> --n 64 \
        --out frame_scaling.json --label Any4D
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from demo_inference import init_inference_model

from any4d.utils.image import load_images
from any4d.utils.inference import loss_of_one_batch_multi_view

CONFIG = {
    "path": "configs/train.yaml",
    "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false",
                         "model/task=images_only"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint_path", default="checkpoints/any4d_4v_combined.pth")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    paths = [str(p) for p in sorted(Path(args.frames).glob("*.jpg"))[:args.n]]
    rec = {"label": args.label, "n": args.n, "amp": "bf16",
           "ckpt": args.checkpoint_path}
    try:
        model = init_inference_model(CONFIG, args.checkpoint_path, torch.device("cuda"))
        torch.cuda.synchronize()
        rec["weights_mib"] = round(torch.cuda.memory_allocated() / 2**20)
        views = load_images(paths, size=(518, 336), norm_type="dinov2", patch_size=14)
        rec["input_hw"] = list(views[0]["img"].shape[-2:])   # the loader keeps the aspect ratio
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            loss_of_one_batch_multi_view(views, model, None, torch.device("cuda"), use_amp=True)
        torch.cuda.synchronize()
        rec.update(status="ok", seconds=round(time.time() - t0, 3))
    except torch.OutOfMemoryError as exc:
        rec.update(status="oom", message=str(exc).splitlines()[0])
    rec.update(torch_peak_alloc_mib=round(torch.cuda.max_memory_allocated() / 2**20),
               torch_peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20))
    out = Path(args.out)
    runs = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps([*runs, rec], indent=2))
    print(f"[probe] {json.dumps(rec)}", flush=True)


if __name__ == "__main__":
    main()
