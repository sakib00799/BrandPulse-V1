import numpy as np
import pandas as pd

from src.evaluation.report import subgroup_metrics


def test_subgroup_metrics_reports_support() -> None:
    rows = subgroup_metrics(
        pd.Series(["a", "a", "b"]),
        np.array(["a", "b", "b"]),
        pd.Series(["x", "x", "y"]),
    )
    assert rows[0]["group"] == "x"
    assert rows[0]["support"] == 2
    assert rows[1]["accuracy"] == 1.0
