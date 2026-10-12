import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_utils

from tools.validate_submission import validate_submission


def test_submission_validation_checks_overlap_and_unknown_images(tmp_path):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))["counts"].decode("utf-8")
    path = tmp_path / "submission.csv"
    rows = [{"filament_id": "test_1", "segmentation_rle": rle}]
    pd.DataFrame(rows).to_csv(path, index=False)
    assert validate_submission(path, {"test"}, shape=mask.shape)["rows"] == 1
    with pytest.raises(ValueError, match="Unknown test"):
        validate_submission(path, {"other"}, shape=mask.shape)
    rows.append({"filament_id": "test_2", "segmentation_rle": rle})
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Overlapping"):
        validate_submission(path, {"test"}, shape=mask.shape)
