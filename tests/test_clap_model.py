from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from audiosr.clap.open_clip.model import CLAP


def _audio_model(model_type, clip_samples, encode_audio):
    model = CLAP.__new__(CLAP)
    torch.nn.Module.__init__(model)
    model.audio_cfg = SimpleNamespace(
        model_type=model_type,
        clip_samples=clip_samples,
    )
    model.encode_audio = Mock(side_effect=encode_audio)
    model.eval()
    return model


def test_audio_infer_squeezes_single_pann_batch():
    model = _audio_model(
        "PANN",
        clip_samples=4,
        encode_audio=lambda audio, device: {
            "embedding": audio[:, :2],
            "fine_grained_embedding": audio[:, :, None],
        },
    )

    output = model.audio_infer(torch.arange(4.0))

    assert output["embedding"].shape == (2,)
    assert output["fine_grained_embedding"].shape == (4, 1)


def test_audio_infer_uses_clip_length_as_default_htsat_hop():
    model = _audio_model(
        "HTSAT",
        clip_samples=4,
        encode_audio=lambda audio, device: {
            "embedding": audio[:, :2],
        },
    )

    output = model.audio_infer(torch.arange(10.0))

    assert output["embedding"].shape == (3, 2)
    encoded_audio = model.encode_audio.call_args.args[0]
    assert encoded_audio.tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0, 9.0],
    ]


@pytest.mark.parametrize("hopsize", [0, -1])
def test_audio_infer_rejects_non_positive_hopsize(hopsize):
    model = _audio_model(
        "HTSAT",
        clip_samples=4,
        encode_audio=lambda audio, device: {"embedding": audio},
    )

    with pytest.raises(ValueError, match="hopsize"):
        model.audio_infer(torch.arange(10.0), hopsize=hopsize)
