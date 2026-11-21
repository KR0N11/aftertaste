from aftertaste.carb_eval import load_cases, run, summarize, to_markdown
from aftertaste.models import Nutrition
from aftertaste.nutrition import ParseError


class FakeParser:
    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    def parse(self, text=None, photo=None):
        self.calls += 1
        value = self.answers[text]
        if value is None:
            raise ParseError("unclear")
        return Nutrition(value, 0, 0, 0, "high" if value > 50 else "medium")


def write_csv(tmp_path):
    f = tmp_path / "labels.csv"
    f.write_text(
        "id,description,photo,true_carbs_g\n"
        "a,apple,,25\nb,pasta,,80\nc,mystery,,40\nd,rice bowl,,60\n"
    )
    return f


def test_scores_and_counts_failures(tmp_path):
    parser = FakeParser({"apple": 20, "pasta": 95, "mystery": None, "rice bowl": 62})
    results = run(load_cases(write_csv(tmp_path)), parser, cache_dir=tmp_path / "cache")
    s = summarize(results)
    assert s["n_cases"] == 4 and s["n_scored"] == 3 and s["n_failed"] == 1
    assert s["mae_g"] == round((5 + 15 + 2) / 3, 1)
    assert s["within_10g_pct"] == round(100 * 2 / 3, 1)
    assert s["worst"][0]["id"] == "b"
    assert "Mean absolute error" in to_markdown(s)


def test_cache_avoids_paying_twice(tmp_path):
    csv_path = write_csv(tmp_path)
    parser = FakeParser({"apple": 20, "pasta": 95, "mystery": None, "rice bowl": 62})
    run(load_cases(csv_path), parser, cache_dir=tmp_path / "cache")
    first = parser.calls
    run(load_cases(csv_path), parser, cache_dir=tmp_path / "cache")
    assert parser.calls - first == 1  # only the failed one is retried


def test_sample_is_fixed_by_seed_and_keeps_order():
    from aftertaste.carb_eval import Case, sample

    cases = [Case(str(i), "", None, float(i)) for i in range(50)]
    a, b = sample(cases, 10, seed=3), sample(cases, 10, seed=3)
    assert [c.id for c in a] == [c.id for c in b]
    assert len(a) == 10 and [int(c.id) for c in a] == sorted(int(c.id) for c in a)
    assert sample(cases, None) == cases
