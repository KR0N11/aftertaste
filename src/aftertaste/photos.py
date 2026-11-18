"""Import meal photos from a folder (e.g. an iCloud Drive folder the phone saves to).

The meal time is the moment the photo was taken (EXIF DateTimeOriginal plus
its UTC offset), not when the file arrived. iPhone HEIC photos are converted
to JPEG and shrunk to at most 1568 px on the long side before the model sees
them, which keeps the request small without losing what is on the plate.
Each file is imported once, tracked by content hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
MAX_SIDE = 1568
EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 36867
OFFSET_TIME_ORIGINAL = 36881


@dataclass(frozen=True)
class Photo:
    source: Path
    sha256: str
    taken_at: datetime
    time_source: str  # "exif" or "file"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def taken_at(path: Path, tz: str = "America/Toronto") -> tuple[datetime, str]:
    """When the photo was taken. Falls back to the file's modification time
    (in `tz`) when the photo has no EXIF date, and says so."""
    with Image.open(path) as img:
        exif = img.getexif().get_ifd(EXIF_IFD)
    raw = exif.get(DATETIME_ORIGINAL)
    if raw:
        when = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
        offset = exif.get(OFFSET_TIME_ORIGINAL)
        if offset:
            return datetime.fromisoformat(when.isoformat() + str(offset).strip()), "exif"
        return when.replace(tzinfo=ZoneInfo(tz)), "exif"
    return datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo(tz)), "file"


def find_new(folder: str | Path, seen: set[str], tz: str = "America/Toronto") -> list[Photo]:
    out = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in EXTENSIONS or path.name.startswith("."):
            continue
        digest = sha256_of(path)
        if digest in seen:
            continue
        when, source = taken_at(path, tz)
        out.append(Photo(path, digest, when, source))
    return sorted(out, key=lambda p: p.taken_at)


def to_model_jpeg(photo: Photo, out_dir: str | Path) -> Path:
    """Upright, at most MAX_SIDE px, JPEG, saved under out_dir by hash."""
    out = Path(out_dir) / f"{photo.sha256[:16]}.jpg"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(photo.source) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(out, "JPEG", quality=88)
    return out
