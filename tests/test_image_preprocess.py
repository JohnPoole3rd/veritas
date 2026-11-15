import numpy as np
import pytest
import torch
from PIL import Image

from src.data import get_transforms
from src.encoders import GPUEvalTransform, UInt8ImageList, collate_by_shape, extract_image_embeddings


def _write(tmp_path, name, size, seed):
    rng = np.random.default_rng(seed)
    path = tmp_path / name
    Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


@pytest.mark.parametrize("size", [(64, 64), (300, 200), (200, 300)])
def test_device_transform_matches_pil_transform(tmp_path, size):
    path = _write(tmp_path, "a.png", size, 0)
    ref = get_transforms(224)(Image.open(path).convert("RGB"))
    _, u8 = UInt8ImageList([path])[0]
    got = GPUEvalTransform(224)(u8[None])[0]
    assert got.shape == ref.shape
    # Upscaling is bit-exact with PIL; antialiased downscaling may differ by a grey level.
    tol = 1e-5 if size == (64, 64) else 0.018
    assert (got - ref).abs().max().item() < tol


class _MeanModel(torch.nn.Module):
    def forward(self, pixel_values):
        class Out:
            pooler_output = pixel_values.float().mean(dim=(2, 3))

        return Out()


def test_device_path_keeps_order_with_mixed_shapes(tmp_path):
    paths = [_write(tmp_path, f"{i}.png", (64, 64) if i % 3 else (80, 96), i) for i in range(7)]
    ref = np.stack([get_transforms(224)(Image.open(p).convert("RGB")).mean(dim=(1, 2)).numpy() for p in paths])
    got = extract_image_embeddings(
        _MeanModel(), UInt8ImageList(paths), batch_size=4, device="cpu", device_transform=GPUEvalTransform(224)
    )
    assert got.shape == (7, 3)
    np.testing.assert_allclose(got, ref, atol=0.02)
    groups = collate_by_shape([UInt8ImageList(paths)[i] for i in range(4)])
    assert sorted(i for idx, _ in groups for i in idx.tolist()) == [0, 1, 2, 3]
