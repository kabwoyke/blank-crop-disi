"""
cropping_core.py
-----------------
Reusable core logic for the QCC Auto-Crop tool.
"""
import base64
import concurrent.futures
import io
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from utils.image_utils import list_images , count_images , BACKUP_DIR_NAME , IMAGE_EXTS
from utils.image_io import save_frames_atomic, copy_unchanged, ensure_output_present
from db import *
import cv2
import numpy as np
from PIL import Image
import platform
from pathlib import Path, PureWindowsPath


logger = logging.getLogger("qcc_autocrop")


# ---------------------------------------------------------------------------
# Core crop-detection logic
# ---------------------------------------------------------------------------

def _detect_crop_box(img_np, threshold):
    """Bounding box of the largest foreground contour, or None."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if w < img_np.shape[1] * 0.4 or h < img_np.shape[0] * 0.4:
        return None

    return x, y, w, h


def _crop_frame(img_pil_frame, threshold):
    """Crop a single PIL frame. Returns (cropped_pil, status, orig_size)."""
    original_mode = img_pil_frame.mode
    rgb_frame = img_pil_frame.convert('RGB') if original_mode != 'RGB' else img_pil_frame.copy()

    img_np = cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR)
    orig_w, orig_h = rgb_frame.width, rgb_frame.height

    box = _detect_crop_box(img_np, threshold)

    if box is not None:
        x, y, w, h = box
        if (w, h) != (orig_w, orig_h):
            img_np = img_np[y:y + h, x:x + w]
            status = 'cropped'
        else:
            status = 'unchanged'
    else:
        status = 'unchanged'

    cropped_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)

    if original_mode in ['1', 'L', 'P', 'RGB', 'RGBA']:
        cropped_pil = cropped_pil.convert(original_mode)

    return cropped_pil, status, (orig_w, orig_h)


def remove_black_borders(image_path, output_path, threshold=150):
    """Crop image_path and save to output_path. Returns (success, message)."""
    try:
        with Image.open(image_path) as img_pil:
            n_frames = getattr(img_pil, "n_frames", 1)
            default_dpi = img_pil.info.get('dpi', (300, 300))
            src_info = dict(img_pil.info)

            cropped_frames, statuses, sizes = [], [], []

            for frame_idx in range(n_frames):
                img_pil.seek(frame_idx)
                cropped_pil, status, orig_size = _crop_frame(img_pil, threshold)
                cropped_frames.append(cropped_pil)
                statuses.append(status)
                sizes.append((orig_size, cropped_pil.size))

            any_cropped = any(s == 'cropped' for s in statuses)

        # Source handle closed -> safe to replace in place on Windows.
        if not any_cropped:
            # Nothing changed: keep the original bytes (no re-encode loss).
            copy_unchanged(image_path, output_path)
            return True, f"UNCHANGED: no reliable border found on any of {n_frames} page(s)"

        save_frames_atomic(cropped_frames, output_path, dpi=default_dpi, src_info=src_info)
        return True, None

    except Exception as e:
        return False, f"Error processing {image_path}: {str(e)}"


def make_thumb_b64(pil_img, max_dim=380):
    """Downscale a PIL image and return it as a base64 JPEG string."""
    thumb = pil_img.copy()
    thumb.thumbnail((max_dim, max_dim))
    if thumb.mode not in ('RGB', 'L'):
        thumb = thumb.convert('RGB')
    buf = io.BytesIO()
    thumb.save(buf, format='JPEG', quality=82)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------





def discover_batches_from_folder(parent_folder):
    """Walk parent_folder/<box>/<batch> and return batch descriptors."""
    parent = Path(parent_folder)
    discovered = []

    if not parent.exists():
        raise FileNotFoundError(f"Folder does not exist: {parent_folder}")
    if not parent.is_dir():
        raise NotADirectoryError(f"Not a folder: {parent_folder}")

    box_dirs = sorted(p for p in parent.iterdir() if p.is_dir())
    if not box_dirs:
        return discovered

    for box_dir in box_dirs:
        batch_dirs = sorted(p for p in box_dir.iterdir() if p.is_dir())
        if batch_dirs:
            for batch_dir in batch_dirs:
                discovered.append({
                    'BatchID': None,
                    'BatchName': f"{box_dir.name}/{batch_dir.name}",
                    'BatchDirectory': str(batch_dir),
                    'ImageCount': count_images(batch_dir),
                })
        else:
            if count_images(box_dir) > 0:
                discovered.append({
                    'BatchID': None,
                    'BatchName': box_dir.name,
                    'BatchDirectory': str(box_dir),
                    'ImageCount': count_images(box_dir),
                })

    return discovered


def backup_target_path(backup_dir, rel):
    """
    Compute the backup file path for `rel` inside `backup_dir`, appending an
    '_1' suffix to the filename (e.g. 'img.jpg' -> 'img_1.jpg').

    If the stem already ends with '_1' (which can happen if a filename
    legitimately already carries that suffix), it is NOT appended again --
    this keeps the naming stable and prevents '_1_1' style collisions when
    multiple pipeline stages (autofill / cropping / orientation) each call
    this helper against the same original filename.
    """
    rel = Path(rel)
    stem = rel.stem
    if stem.endswith('_1'):
        new_name = rel.name
    else:
        new_name = f"{stem}_1{rel.suffix}"
    return backup_dir / rel.with_name(new_name)


def ensure_backup_folder(batch_dir):
    """Create QCCBackups under batch_dir. Skip (no-op) if it already exists."""
    backup_dir = Path(batch_dir) / BACKUP_DIR_NAME
    if backup_dir.exists():
        return {'created': False, 'path': str(backup_dir), 'status': 'skipped (already exists)'}
    backup_dir.mkdir(parents=True, exist_ok=True)
    return {'created': True, 'path': str(backup_dir), 'status': 'created'}


def prepare_backup_folders(batches):
    """For every batch that has images, ensure its QCCBackups folder exists."""
    results = []
    for b in batches:
        batch_dir = Path(b['BatchDirectory'])
        if not batch_dir.exists():
            results.append({
                'batch_name': b['BatchName'], 'batch_directory': str(batch_dir),
                'status': 'skipped (directory missing)', 'image_count': 0,
            })
            continue
        n_images = count_images(batch_dir)
        if n_images == 0:
            results.append({
                'batch_name': b['BatchName'], 'batch_directory': str(batch_dir),
                'status': 'skipped (no images)', 'image_count': 0,
            })
            continue
        r = ensure_backup_folder(batch_dir)
        results.append({
            'batch_name': b['BatchName'],
            'batch_directory': str(batch_dir),
            'status': r['status'],
            'backup_path': r['path'],
            'image_count': n_images,
        })
    return results


# ---------------------------------------------------------------------------
# Preview (read-only)
# ---------------------------------------------------------------------------

def generate_preview(batch_dir, threshold=150, sample_size=6):
    batch_dir = Path(batch_dir)
    images = list_images(batch_dir)
    sample = images[:max(0, sample_size)]
    previews = []

    for p in sample:
        try:
            with Image.open(p) as img:
                img.seek(0)
                frame = img.convert('RGB') if img.mode != 'RGB' else img.copy()
                cropped_pil, status, orig_size = _crop_frame(frame, threshold)
                previews.append({
                    'filename': p.name,
                    'status': status,
                    'orig_size': list(orig_size),
                    'new_size': list(cropped_pil.size),
                    'before': make_thumb_b64(frame),
                    'after': make_thumb_b64(cropped_pil),
                })
        except Exception as e:
            previews.append({'filename': p.name, 'status': 'error', 'error': str(e)})

    return {
        'batch_directory': str(batch_dir),
        'total_images': len(images),
        'sample_count': len(sample),
        'previews': previews,
    }


# ---------------------------------------------------------------------------
# Live run: backup-then-crop
# ---------------------------------------------------------------------------

def process_batch_with_backup(batch_dir, threshold=150, progress_cb=None, cancel_check=None,
                              skip_backup=False, workers=None):
    """
    Crop every image in batch_dir.

    skip_backup=True: this stage is part of a multi-stage chain and an earlier
    stage has already moved originals to QCCBackups.  Read directly from img_path
    and write in place; skip all backup accounting.

    workers: size of the thread pool used to crop images in parallel. cv2/PIL
    release the GIL for the bulk of their work (decode, contour detection,
    encode), so this parallelizes well across threads. If None, falls back to
    a sensible CPU-based default. Pass config_store.get_workers('crop') from
    the caller to let the user tune this from settings.
    """
    batch_dir = Path(batch_dir)
    backup_dir = batch_dir / BACKUP_DIR_NAME

    if not skip_backup:
        backup_dir.mkdir(parents=True, exist_ok=True)

    scan_start = time.perf_counter()
    image_files = list_images(batch_dir)
    total = len(image_files)
    logger.info(
        "[crop] batch=%s discovered %d images in %.2fs",
        batch_dir.name, total, time.perf_counter() - scan_start,
    )

    stats = {
        'total_images': total,
        'moved_to_backup': 0,
        'already_backed_up': 0,
        'cropped_success': 0,
        'cropped_unchanged': 0,
        'cropped_failed': 0,
        'backup_dir': str(backup_dir),
        'errors': [],
    }

    if total == 0:
        return stats

    if progress_cb:
        progress_cb(0, total, "")

    lock = threading.Lock()
    completed_count = 0

    def _process_one(item):
        idx, img_path = item
        if cancel_check and cancel_check():
            return idx, None, "cancelled", "Cancelled by user", None, None

        rel = img_path.relative_to(batch_dir)
        t0 = time.perf_counter()
        try:
            if skip_backup:
                source_path = img_path
                moved, already = False, False
            else:
                backup_path = backup_target_path(backup_dir, rel)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                if backup_path.exists():
                    source_path = img_path
                    moved, already = False, True
                else:
                    shutil.move(str(img_path), str(backup_path))
                    source_path = backup_path
                    moved, already = True, False

            success, msg = remove_black_borders(str(source_path), str(img_path), threshold=threshold)
            if not success:
                ensure_output_present(source_path, img_path)
            logger.debug(
                "[crop][%s] %s -> %s (%.2fs)",
                threading.current_thread().name, rel, msg, time.perf_counter() - t0,
            )
            return idx, success, msg, None, rel, (moved, already)
        except Exception as e:
            logger.debug(
                "[crop][%s] %s -> error (%.2fs): %s",
                threading.current_thread().name, rel, time.perf_counter() - t0, e,
            )
            try:
                ensure_output_present(source_path, img_path)
            except NameError:
                pass
            return idx, False, str(e), None, rel, (False, False)

    max_workers = workers if workers else min(16, max(4, (os.cpu_count() or 4) * 2))
    max_workers = max(1, int(max_workers))
    batch_start = time.perf_counter()
    log_every = max(1, total // 10)

    logger.info(
        "[crop] batch=%s images=%d workers=%d",
        batch_dir.name, total, max_workers,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="crop-worker") as executor:
        futures = [executor.submit(_process_one, (i, p)) for i, p in enumerate(image_files, start=1)]
        for f in concurrent.futures.as_completed(futures):
            idx, success, msg, _unused, rel, backup_flags = f.result()
            completed_count += 1

            if success is None:
                # cancelled before doing any work
                stats['cancelled_at'] = idx
                if progress_cb and rel:
                    progress_cb(completed_count, total, str(rel))
                continue

            if backup_flags:
                moved, already = backup_flags
                if moved:
                    stats['moved_to_backup'] += 1
                if already:
                    stats['already_backed_up'] += 1

            if not success:
                stats['cropped_failed'] += 1
                stats['errors'].append(f"{rel}: {msg}")
            elif msg and msg.startswith("UNCHANGED:"):
                stats['cropped_unchanged'] += 1
            else:
                stats['cropped_success'] += 1

            if completed_count % log_every == 0 or completed_count == total:
                elapsed = time.perf_counter() - batch_start
                rate = completed_count / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[crop] batch=%s %d/%d done (%.1f img/s, %d workers)",
                    batch_dir.name, completed_count, total, rate, max_workers,
                )

            if progress_cb and rel:
                progress_cb(completed_count, total, str(rel))

    total_elapsed = time.perf_counter() - batch_start
    avg_rate = completed_count / total_elapsed if total_elapsed > 0 else 0.0
    logger.info(
        "[crop] batch=%s finished: %d/%d images in %.1fs (%.1f img/s avg, %d workers) -- "
        "success=%d unchanged=%d failed=%d",
        batch_dir.name, completed_count, total, total_elapsed, avg_rate, max_workers,
        stats['cropped_success'], stats['cropped_unchanged'], stats['cropped_failed'],
    )

    return stats


# ---------------------------------------------------------------------------
# SQL Server helpers with status column support
# ---------------------------------------------------------------------------

from db import _connect



def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


# RESTORATION
# ---------------------------------------------------------------------------
# Restore: move backed-up originals back over the processed images
# ---------------------------------------------------------------------------

def original_path_from_backup(batch_dir, backup_rel):
    """
    Inverse of backup_target_path().

    Given a path relative to backup_dir (e.g. 'img_1.jpg'), return the
    corresponding path in batch_dir with the '_1' suffix stripped
    (e.g. batch_dir/'img.jpg').

    If the backup filename does NOT end in '_1' (edge case: the original
    file's stem already ended in '_1' before backup, so backup_target_path
    left it unchanged), the name is used as-is.
    """
    rel = Path(backup_rel)
    stem = rel.stem
    if stem.endswith('_1'):
        orig_stem = stem[:-2]
    else:
        orig_stem = stem
    return Path(batch_dir) / rel.with_name(f"{orig_stem}{rel.suffix}")



# ---------------------------------------------------------------------------
# Restore a single file from QCCBackups back to the batch root
# ---------------------------------------------------------------------------

def restore_single_file(batch_dir, filename):
    """
    Move one file from batch_dir/QCCBackups back to batch_dir, restoring the
    original (pre-processing) version and overwriting the current processed
    version sitting at that path. QCCBackups itself is never deleted --
    only the one matching file is moved out of it.
    """
    batch_dir = Path(batch_dir)
    backup_dir = batch_dir / BACKUP_DIR_NAME

    if not backup_dir.exists():
        return {'success': False, 'error': f"No backup folder found at {backup_dir}"}

    # filename passed in is the ORIGINAL name (e.g. 'img.jpg'), so we need to
    # find its backed-up counterpart inside QCCBackups (e.g. 'img_1.jpg')
    target_path = batch_dir / filename
    backup_path = backup_target_path(backup_dir, filename)

    if not backup_path.exists():
        return {
            'success': False,
            'error': f"Backup file not found: {backup_path}",
        }

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(backup_path), str(target_path))
    except Exception as e:
        return {'success': False, 'error': str(e)}

    return {
        'success': True,
        'restored_file': str(target_path),
        'backup_path_removed': str(backup_path),
        'backup_dir': str(backup_dir),
    }


def find_backup_file(backup_dir, filename):
    """
    Locate the actual backup file for `filename` inside backup_dir, regardless
    of whether it was stored with the '_1' suffix convention or under its
    plain original name (some backups are intentionally stored without the
    suffix). Checks both, in order:

      1. backup_dir/<stem>_1<suffix>   (e.g. 'image_1.jpg' for 'image.jpg')
      2. backup_dir/<filename>          (e.g. 'image.jpg' for 'image.jpg')

    Returns the first Path that actually exists, or None if neither does.
    """
    backup_dir = Path(backup_dir)

    suffixed = backup_target_path(backup_dir, filename)
    if suffixed.exists():
        return suffixed

    plain = backup_dir / filename
    if plain.exists():
        return plain

    return None


def restore_single_file(batch_dir, filename):
    """
    Move one file from batch_dir/QCCBackups back to batch_dir, restoring the
    original (pre-processing) version and overwriting the current processed
    version sitting at that path. QCCBackups itself is never deleted --
    only the one matching file is moved out of it.

    Handles both backup naming conventions: 'image_1.jpg' and plain
    'image.jpg' (some backups are stored without the '_1' suffix by design).
    """
    batch_dir = Path(batch_dir)
    backup_dir = batch_dir / BACKUP_DIR_NAME

    if not backup_dir.exists():
        return {'success': False, 'error': f"No backup folder found at {backup_dir}"}

    target_path = batch_dir / filename
    backup_path = find_backup_file(backup_dir, filename)

    if backup_path is None:
        return {
            'success': False,
            'error': f"Backup file not found for '{filename}' in {backup_dir}",
            'checked_paths': [
                str(backup_target_path(backup_dir, filename)),
                str(backup_dir / filename),
            ],
        }

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(backup_path), str(target_path))
    except Exception as e:
        return {'success': False, 'error': str(e)}

    return {
        'success': True,
        'restored_file': str(target_path),
        'backup_path_used': str(backup_path),
        'backup_filename': backup_path.name,
        'backup_dir': str(backup_dir),
    }


# def restore_batch_from_backup(batch_dir, progress_cb=None, cancel_check=None,
#                                remove_backup_dir=False):
#     """
#     Restore every image in batch_dir/QCCBackups back to its original location,
#     overwriting whatever processed (cropped/rotated/filled) version currently
#     sits there.

#     remove_backup_dir=True: delete the now-empty QCCBackups folder once every
#     file has been moved back out of it. Leave False if you want to re-run the
#     pipeline again afterwards without re-discovering the batch.
#     """
#     batch_dir = Path(batch_dir)
#     backup_dir = batch_dir / BACKUP_DIR_NAME

#     stats = {
#         'total_backup_images': 0,
#         'restored': 0,
#         'failed': 0,
#         'errors': [],
#         'backup_dir': str(backup_dir),
#     }

#     if not backup_dir.exists():
#         stats['error'] = f"No backup folder found at {backup_dir}"
#         return stats

#     backup_files = list_images(backup_dir)
#     stats['total_backup_images'] = len(backup_files)
#     total = len(backup_files)

#     for idx, backup_path in enumerate(backup_files, start=1):
#         if cancel_check and cancel_check():
#             stats['cancelled_at'] = idx
#             break

#         rel = backup_path.relative_to(backup_dir)
#         target_path = original_path_from_backup(batch_dir, rel)

#         try:
#             target_path.parent.mkdir(parents=True, exist_ok=True)
#             shutil.move(str(backup_path), str(target_path))
#             stats['restored'] += 1
#         except Exception as e:
#             stats['failed'] += 1
#             stats['errors'].append(f"{rel}: {str(e)}")

#         if progress_cb:
#             progress_cb(idx, total, str(rel))

#     if remove_backup_dir:
#         try:
#             # only rmtree if it's actually empty of images to avoid nuking
#             # anything unexpected that landed in there
#             remaining = list_images(backup_dir)
#             if not remaining:
#                 shutil.rmtree(backup_dir)
#                 stats['backup_dir_removed'] = True
#             else:
#                 stats['backup_dir_removed'] = False
#                 stats['backup_dir_removed_reason'] = f"{len(remaining)} files still present"
#         except Exception as e:
#             stats['backup_dir_removed'] = False
#             stats['remove_error'] = str(e)

#     return stats