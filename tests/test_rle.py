import numpy as np

from solar.rle import MASK_SIZE, decode_rle_counts, encode_binary_mask


def test_rle_roundtrip() -> None:
    mask = np.zeros(MASK_SIZE, dtype=bool)
    mask[10:20, 30:50] = True

    counts = encode_binary_mask(mask)
    decoded = decode_rle_counts(counts)

    assert decoded.shape == MASK_SIZE
    assert np.array_equal(decoded, mask)

