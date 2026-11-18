from datetime import timezone
from pathlib import Path

import pytest

from aftertaste.clarity import import_clarity_csv
from aftertaste.models import InsulinKind

FIXTURE = Path(__file__).parent / "fixtures" / "clarity_mmol.csv"


def test_imports_mmol_export_with_local_time():
    readings, doses = import_clarity_csv(FIXTURE, tz="America/Toronto")
    assert [round(r.mgdl) for r in readings] == [110, 40, 400]
    # 12:00 local in March before DST = 17:00 UTC
    assert readings[0].time.utcoffset().total_seconds() == -5 * 3600
    assert readings[0].time.astimezone(timezone.utc).hour == 17
    assert [(d.units, d.kind) for d in doses] == [(4.5, InsulinKind.RAPID), (18.0, InsulinKind.LONG)]


def test_rejects_other_csvs(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not a Clarity export"):
        import_clarity_csv(f)
