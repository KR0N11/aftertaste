import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from aftertaste import service
from aftertaste.models import Nutrition
from aftertaste.nutrition import ParseError
from aftertaste.photos import find_new, taken_at, to_model_jpeg
from aftertaste.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "clarity_mmol.csv"


def photo(path: Path, when: str | None = "2026:03:02 12:00:00", offset: str | None = "-05:00",
          size=(4032, 3024), fmt="JPEG"):
    img = Image.new("RGB", size, (200, 120, 40))
    exif = Image.Exif()
    if when:
        ifd = exif.get_ifd(0x8769)
        ifd[36867] = when
        if offset:
            ifd[36881] = offset
    img.save(path, fmt, exif=exif)
    return path


class FakeParser:
    def parse(self, text=None, photo=None):
        return Nutrition(55, 4, 12, 20, "medium", ("rice bowl (1 bowl)",))


def test_exif_time_with_offset(tmp_path):
    when, source = taken_at(photo(tmp_path / "a.jpg"))
    assert source == "exif"
    assert when.astimezone(timezone.utc) == datetime(2026, 3, 2, 17, 0, tzinfo=timezone.utc)


def test_falls_back_to_file_time_and_says_so(tmp_path):
    _, source = taken_at(photo(tmp_path / "b.jpg", when=None))
    assert source == "file"


def test_heic_is_read_and_converted_small(tmp_path):
    p = photo(tmp_path / "c.heic", fmt="HEIF")
    [found] = find_new(tmp_path, set())
    out = to_model_jpeg(found, tmp_path / "out")
    with Image.open(out) as img:
        assert img.format == "JPEG" and max(img.size) <= 1568


def test_daily_imports_once_and_retries_failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    meals_dir = tmp_path / "Meals"
    meals_dir.mkdir()
    photo(meals_dir / "lunch.jpg")
    photo(meals_dir / "dinner.jpg", when="2026:03:02 19:00:00", size=(800, 600))
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    shutil.copy(FIXTURE, downloads / "export.csv")
    (downloads / "other.csv").write_text("a,b\n1,2\n")

    store = Store(tmp_path / "t.db")
    first = service.daily(store, _FailSecondCall(FakeParser()), meals_dir, downloads)
    assert len(first["photos"]["imported"]) == 1
    assert len(first["photos"]["failed"]) == 1
    assert first["clarity"] == {"files": 1, "readings": 3, "insulin_doses": 2}

    second = service.daily(store, FakeParser(), meals_dir, downloads)
    assert [p["file"] for p in second["photos"]["imported"]] == ["dinner.jpg"]  # retried
    assert second["clarity"]["files"] == 0  # same export not re-imported
    third = service.daily(store, FakeParser(), meals_dir, downloads)
    assert third["photos"]["imported"] == [] and len(store.meals()) == 2


class _FailSecondCall:
    """Photos are processed in time order (lunch, then dinner): fail dinner."""

    def __init__(self, parser):
        self.parser, self.calls = parser, 0

    def parse(self, text=None, photo=None):
        self.calls += 1
        if self.calls == 2:
            raise ParseError("blurry")
        return self.parser.parse(text, photo)


def test_unreadable_folder_is_reported_not_silent(tmp_path, monkeypatch):
    import os

    monkeypatch.chdir(tmp_path)
    real = os.listdir

    def deny(path):
        if "Meals" in str(path):
            raise PermissionError("Operation not permitted")
        return real(path)

    monkeypatch.setattr(os, "listdir", deny)
    (tmp_path / "Meals").mkdir()
    report = service.daily(Store(tmp_path / "t.db"), FakeParser(), tmp_path / "Meals", tmp_path)
    assert any("Full Disk Access" in p for p in report["problems"])
