import pandas as pd
import pytest

from src.profiling_utils import feature_completeness_report


def test_feature_completeness_uses_row_level_union():
    frame = pd.DataFrame({
        "a": [1, None, 3, 4],
        "b": [1, 2, None, 4],
        "target": [0, 1, 1, 0],
    })

    report = feature_completeness_report(frame, ["a", "b"])

    assert report["total_rows"] == 4
    assert report["complete_rows"] == 2
    assert report["incomplete_rows"] == 2
    assert report["complete_pct"] == 50.0
    assert report["incomplete_pct"] == 50.0
    assert report["per_feature"]["missing_count"].tolist() == [1, 1]


def test_feature_completeness_rejects_unknown_or_empty_features():
    frame = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="missing from dataframe"):
        feature_completeness_report(frame, ["unknown"])

    with pytest.raises(ValueError, match="at least one"):
        feature_completeness_report(frame, [])
