#!/usr/bin/env python3
import os
import sys
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageSequence

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "frames")

TARGETS: Dict[str, str] = {
    "circle-left": "circle",
    "circle-right": "circle",
    "no_gesture": "no_gesture",
    "open-palm-left": "open_palm",
    "open-palm-right": "open_palm",
    "pinch-and-drag-left-to-right-left": "pinch_drag",
    "pinch-and-drag-left-to-right-right": "pinch_drag",
    "pinch-and-drag-right-to-left-left": "pinch_drag",
    "pinch-and-drag-right-to-left-right": "pinch_drag",
    "thumbs-up-left": "thumb_up",
    "thumbs-up-right": "thumb_up",
    "thumbs-down-left": "thumb_down",
    "thumbs-down-right": "thumb_down",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_gif_path(base_name: str) -> Optional[str]:
    candidates: List[str] = []
    candidates.append(os.path.join(ASSETS_DIR, f"{base_name}.gif"))
    candidates.append(os.path.join(ASSETS_DIR, f"{base_name.replace('-', '_')}.gif"))
    candidates.append(os.path.join(ASSETS_DIR, f"{base_name.replace('_', '-')}.gif"))
    candidates.append(os.path.join(ASSETS_DIR, f"{base_name.lower()}.gif"))
    candidates.append(os.path.join(ASSETS_DIR, f"{base_name.upper()}.gif"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def extract_frames(gif_path: str, out_dir: str, prefix: str) -> int:
    ensure_dir(out_dir)
    count = 0
    with Image.open(gif_path) as im:
        for idx, frame in enumerate(ImageSequence.Iterator(im)):
            rgb = frame.convert("RGB")
            out_path = os.path.join(out_dir, f"{prefix}_frame{idx:04d}.png")
            rgb.save(out_path, format="PNG")
            count += 1
    return count


def main() -> int:
    out_root = DEFAULT_OUT_DIR
    ensure_dir(out_root)
    total = 0
    missing: List[str] = []
    for base, label in TARGETS.items():
        gif_path = find_gif_path(base)
        if not gif_path:
            missing.append(base)
            continue
        label_dir = os.path.join(out_root, label)
        written = extract_frames(gif_path, label_dir, prefix=os.path.splitext(os.path.basename(gif_path))[0])
        print(f"[OK] {base}.gif -> {label} ({written} frames)")
        total += written
    if missing:
        print(f"[WARN] Missing GIFs (not found in assets/): {', '.join(missing)}")
    print(f"Done. Total frames written: {total}. Output root: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



