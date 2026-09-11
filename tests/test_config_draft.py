from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from src.profiling_utils import (
    draft_data_config,
    summarize_unresolved,
    write_data_config,
)


def make_draft_frame():
    return pd.DataFrame({
        "income": [100, 200, 300],
        "category": ["a", "b", "a"],
        "loan_status": [0, 1, 1],
    })


def test_draft_contains_baseline_and_final_spec_placeholders():
    draft = draft_data_config(
        make_draft_frame(),
        target_column="loan_status",
        dataset_name="draft-test",
    )

    review = draft["needs_review"]
    assert "BASELINE_MODEL_SPECS" in review
    assert "MODEL_SPECS" in review
    assert set(review["BASELINE_MODEL_SPECS"]) == {"classification", "regression"}
    assert set(review["MODEL_SPECS"]) == {"classification", "regression"}

    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        output_path = Path(temp_dir) / "data_config_draft.py"
        content = write_data_config(draft, output_path)
    assert "BASELINE_MODEL_SPECS" in content
    assert "MODEL_SPECS" in content

    unresolved = summarize_unresolved(draft)
    assert set(["BASELINE_MODEL_SPECS", "MODEL_SPECS"]).issubset(
        unresolved["key"]
    )


def test_curated_data_config_is_not_overwritten():
    draft = draft_data_config(
        make_draft_frame(),
        target_column="loan_status",
        dataset_name="draft-test",
    )
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        output_path = Path(temp_dir) / "data_config.py"
        output_path.write_text("# curated configuration\n", encoding="utf-8")

        with pytest.raises(FileExistsError):
            write_data_config(draft, output_path)
