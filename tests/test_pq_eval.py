import pandas as pd

from solar.pq_eval import get_overlap_df, pq_score
from solar.rle import MASK_SIZE, encode_binary_mask


def test_pq_perfect_match() -> None:
    import numpy as np

    mask = np.zeros(MASK_SIZE, dtype=bool)
    mask[50:80, 90:130] = True
    rle = encode_binary_mask(mask)

    gt_df = pd.DataFrame([{"filament_id": "ann1-image_a_1", "segmentation_rle": rle}])
    pred_df = pd.DataFrame([{"filament_id": "image_a_1", "segmentation_rle": rle}])

    overlap = get_overlap_df(gt_df, pred_df)

    assert pq_score(overlap) == 1.0
