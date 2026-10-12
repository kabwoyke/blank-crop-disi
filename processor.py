"""
processor.py
------------
Backup -> auto-crop -> blank-page removal for a folder of images.

Flow per folder:
  1. Copy every original into <folder>/backup (never overwrites an existing backup).
  2. Crop away the dark border around the page (logic from cropping_core.py).
  3. If the cropped page is blank, remove it from the folder (the copy in
     backup remains). Multi-page TIFFs drop only their blank frames.
"""
import concurrent.futures
import os
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BACKUP_DIR_NAME = "backup"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_crop_box(img_bgr, threshold):
    """Bounding box of the largest foreground contour, or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < img_bgr.shape[1] * 0.4 or h < img_bgr.shape[0] * 0.4:
        return None
    return x, y, w, h


def is_blank(img_bgr, sensitivity=0.1, margin=0.03):
    """
    True if the page has (almost) no content.

    The outer `margin` fraction is ignored (scan edges, shadows, punch holes).
    A pixel counts as content if it differs from the page's background level
    (median grey) by more than 40. The page is blank when content pixels make
    up less than `sensitivity` percent of the area (0.1 -> 0.1%).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    my, mx = int(h * margin), int(w * margin)
    inner = gray[my:h - my, mx:w - mx] if h > 2 * my + 1 and w > 2 * mx + 1 else gray
    # Light blur so isolated scanner speckle doesn't count as content.
    inner = cv2.GaussianBlur(inner, (3, 3), 0)
    background = float(np.median(inner))
    content = np.abs(inner.astype(np.int16) - background) > 40
    return content.mean() * 100.0 < sensitivity


def _process_frame(frame, threshold, sensitivity):
    """Returns (cropped_pil, was_cropped, blank)."""
    mode = frame.mode
    rgb = frame.convert("RGB") if mode != "RGB" else frame.copy()
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    cropped = False
    box = _detect_crop_box(bgr, threshold)
    if box is not None and (box[2], box[3]) != (w, h):
        x, y, bw, bh = box
        bgr = bgr[y:y + bh, x:x + bw]
        cropped = True

    blank = is_blank(bgr, sensitivity)
    out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if mode in ("1", "L", "P", "RGB", "RGBA"):
        out = out.convert(mode)
    return out, cropped, blank


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def list_images(folder, recursive=False):
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    out = []
    for p in it:
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if BACKUP_DIR_NAME in p.relative_to(folder).parts[:-1]:
            continue
        out.append(p)
    return sorted(out)


def _save(frames, path, dpi, ext):
    """Write to a temp file then atomically replace the target."""
    tmp = path.with_name(path.stem + ".__tmp__" + path.suffix)
    kw = {}
    if dpi:
        kw["dpi"] = dpi
    if ext in (".jpg", ".jpeg"):
        frames = [f.convert("RGB") if f.mode not in ("RGB", "L") else f for f in frames]
        kw.update(quality=95, subsampling=0)
    if len(frames) > 1 and ext in (".tif", ".tiff"):
        kw.update(save_all=True, append_images=frames[1:], compression="tiff_lzw")
    elif ext in (".tif", ".tiff"):
        kw["compression"] = "tiff_lzw"
    try:
        frames[0].save(tmp, **kw)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def process_image(path, folder, backup_dir, threshold, sensitivity, remove_blank, do_crop):
    """Backup then process one image. Returns a result dict."""
    rel = path.relative_to(folder)
    backup_path = backup_dir / rel
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():  # keep the true original from the first run
        shutil.copy2(path, backup_path)

    # Always read from the backup so reruns start from the original.
    with Image.open(backup_path) as img:
        n = getattr(img, "n_frames", 1)
        dpi = img.info.get("dpi")
        kept, any_cropped, blanks = [], False, 0
        for i in range(n):
            img.seek(i)
            frame, cropped, blank = _process_frame(img, threshold if do_crop else 0, sensitivity)
            any_cropped |= cropped
            if blank and remove_blank:
                blanks += 1
            else:
                kept.append(frame)

    if not kept:
        path.unlink()
        return {"file": str(rel), "status": "blank_removed"}

    if any_cropped or blanks:
        _save(kept, path, dpi, path.suffix.lower())
        status = "cropped" if any_cropped else "unchanged"
        if blanks:
            status += f" (+{blanks} blank page(s) dropped)"
        return {"file": str(rel), "status": status}

    if path.read_bytes() != backup_path.read_bytes():
        shutil.copy2(backup_path, path)
    return {"file": str(rel), "status": "unchanged"}


def process_folder(folder, threshold=150, sensitivity=0.1, remove_blank=True,
                   do_crop=True, recursive=False, progress_cb=None, cancel_check=None):
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    images = list_images(folder, recursive)
    total = len(images)
    stats = {"total": total, "cropped": 0, "unchanged": 0, "blank_removed": 0,
             "failed": 0, "errors": [], "backup_dir": str(folder / BACKUP_DIR_NAME)}
    if total == 0:
        return stats

    backup_dir = folder / BACKUP_DIR_NAME
    backup_dir.mkdir(exist_ok=True)

    lock = threading.Lock()
    done = 0

    def work(p):
        if cancel_check and cancel_check():
            return p, None, "cancelled"
        try:
            return p, process_image(p, folder, backup_dir, threshold, sensitivity,
                                    remove_blank, do_crop), None
        except Exception as e:
            return p, None, str(e)

    workers = min(8, max(1, (os.cpu_count() or 2)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(work, p) for p in images]):
            p, res, err = fut.result()
            with lock:
                done += 1
                if err == "cancelled":
                    stats["cancelled"] = True
                elif err:
                    stats["failed"] += 1
                    stats["errors"].append(f"{p.name}: {err}")
                elif res["status"] == "blank_removed":
                    stats["blank_removed"] += 1
                elif res["status"].startswith("cropped"):
                    stats["cropped"] += 1
                else:
                    stats["unchanged"] += 1
            if progress_cb:
                progress_cb(done, total, p.name if not err else f"{p.name} (error)")
    return stats
