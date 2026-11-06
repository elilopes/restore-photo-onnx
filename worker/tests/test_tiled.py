import numpy as np

from app.ml.tiled import tile_starts, tiled_apply


def test_tile_starts_cover_the_whole_length():
    for length, tile, overlap in [(100, 64, 16), (64, 64, 16), (50, 64, 16), (257, 64, 8)]:
        starts = tile_starts(length, tile, overlap)
        assert starts[0] == 0
        if length > tile:
            assert starts[-1] + tile == length


def test_identity_reconstructs_image_exactly_with_overlaps():
    rng = np.random.default_rng(1)
    img = rng.random((150, 170, 3), dtype=np.float32)
    out, covered = tiled_apply(img, tile=64, overlap=16, scale=1, fn=lambda t, a: t)
    assert covered.all()
    assert np.allclose(out, img, atol=1e-5)


def test_scale_2_matches_nearest_upsample():
    rng = np.random.default_rng(2)
    img = rng.random((90, 70, 3), dtype=np.float32)
    fn = lambda t, a: np.repeat(np.repeat(t, 2, axis=0), 2, axis=1)
    out, _ = tiled_apply(img, tile=32, overlap=8, scale=2, fn=fn)
    expected = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    assert out.shape == expected.shape
    assert np.allclose(out, expected, atol=1e-5)


def test_image_smaller_than_tile_is_padded_and_cropped_back():
    img = np.random.default_rng(3).random((20, 30, 3), dtype=np.float32)
    out, covered = tiled_apply(img, tile=64, overlap=16, scale=1, fn=lambda t, a: t)
    assert out.shape == img.shape and covered.all()
    assert np.allclose(out, img, atol=1e-5)


def test_blocks_without_mask_are_skipped():
    img = np.zeros((200, 200, 3), np.float32)
    mask = np.zeros((200, 200), np.float32)
    mask[5:10, 5:10] = 1  # só o bloco do canto superior esquerdo
    calls = []

    def fn(t, a):
        calls.append(1)
        return t

    out, covered = tiled_apply(img, tile=64, overlap=8, scale=1, fn=fn, aux=mask, skip_empty_aux=True)
    assert len(calls) == 1
    assert covered[:64, :64].all() and not covered[100:, 100:].any()


def test_no_block_processed_returns_none():
    img = np.zeros((64, 64, 3), np.float32)
    mask = np.zeros((64, 64), np.float32)
    assert tiled_apply(img, 32, 8, 1, lambda t, a: t, aux=mask, skip_empty_aux=True) is None
