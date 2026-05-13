"""Tests for the EnrollmentEncoder branch + NanoTSE wiring.

EnrollmentEncoder is the design-fix that gives the TSE model a real cue
about *which* speaker to extract (otherwise the 2-speaker mix is
ambiguous). These tests verify shape contracts, the four cue combinations
(visual on/off x enrollment on/off), and end-to-end gradient flow when
both cues are active.
"""

from __future__ import annotations

import pytest
import torch

from nanotse.data import VoxCeleb2MixDataset
from nanotse.models.frontends.enrollment import EnrollmentEncoder
from nanotse.models.nanotse import NanoTSE


def test_enrollment_encoder_shape() -> None:
    enc = EnrollmentEncoder(d_enroll=128)
    audio = torch.randn(3, 16000)
    out = enc(audio)
    assert out.shape == (3, 128)


def test_enrollment_encoder_rejects_3d_input() -> None:
    enc = EnrollmentEncoder()
    with pytest.raises(ValueError, match="2D"):
        enc(torch.randn(2, 1, 16000))


def test_nanotse_with_enrollment_only_returns_audio_only_shape() -> None:
    """with_visual=False, with_enrollment=True: audio-only path + enrollment cue."""
    m = NanoTSE(
        d_model=32,
        n_heads=2,
        n_layers=1,
        with_visual=False,
        with_enrollment=True,
        d_enroll=64,
    )
    audio = torch.randn(2, 8000)
    enroll = torch.randn(2, 8000)
    tse_out, asd, slots = m(audio, video=None, enrollment=enroll)
    assert tse_out.shape == (2, 8000)
    assert asd is None
    assert slots is None


def test_nanotse_with_visual_and_enrollment_full_av_path() -> None:
    """Both cues active: full AV path + enrollment broadcast into encoder."""
    m = NanoTSE(
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_visual=32,
        n_slots=4,
        d_slot=32,
        frame_size=32,
        with_enrollment=True,
        d_enroll=64,
    )
    audio = torch.randn(1, 16000)
    video = torch.randint(0, 256, (1, 25, 32, 32, 3), dtype=torch.uint8)
    enroll = torch.randn(1, 12000)
    tse_out, asd_logits, slots = m(audio, video, enroll)
    assert tse_out.shape == (1, 16000)
    assert asd_logits is not None
    assert slots is not None


def test_nanotse_enrollment_gradients_flow_to_encoder() -> None:
    """Enrollment must produce gradients on the EnrollmentEncoder params."""
    torch.manual_seed(0)
    m = NanoTSE(
        d_model=32,
        n_heads=2,
        n_layers=1,
        with_visual=False,
        with_enrollment=True,
        d_enroll=64,
    )
    audio = torch.randn(1, 8000)
    enroll = torch.randn(1, 8000)
    tse_out, _, _ = m(audio, enrollment=enroll)
    tse_out.sum().backward()
    grads = [p.grad for p in m.enrollment_encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads if g is not None)


def test_nanotse_enrollment_changes_output_vs_no_enrollment() -> None:
    """Same mix, different enrollment => different output. Identifiability check."""
    torch.manual_seed(0)
    m = NanoTSE(
        d_model=32,
        n_heads=2,
        n_layers=1,
        with_visual=False,
        with_enrollment=True,
        d_enroll=64,
    )
    m.eval()
    audio = torch.randn(1, 8000)
    enroll_a = torch.randn(1, 8000, generator=torch.Generator().manual_seed(1))
    enroll_b = torch.randn(1, 8000, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        out_a, _, _ = m(audio, enrollment=enroll_a)
        out_b, _, _ = m(audio, enrollment=enroll_b)
    assert not torch.allclose(out_a, out_b, atol=1e-5), (
        "enrollment is not influencing the output; conditioning is broken"
    )


def test_voxceleb2_loader_returns_distinct_enrollment_clip(tmp_path) -> None:
    """Real loader must pick a *different* clip from the same speaker as enrollment."""
    import json

    import numpy as np
    import soundfile as sf

    audio_dir = tmp_path / "audio" / "id00001"
    audio_dir.mkdir(parents=True)
    # Each clip MUST have distinct samples or the test can't detect a different-clip pick.
    for i, clip in enumerate(("00001.wav", "00002.wav", "00003.wav")):
        sf.write(
            audio_dir / clip,
            np.random.default_rng(100 + i).standard_normal(16000).astype(np.float32),
            16000,
        )
    audio_b = tmp_path / "audio" / "id00002"
    audio_b.mkdir(parents=True)
    for i, clip in enumerate(("00001.wav", "00002.wav")):
        sf.write(
            audio_b / clip,
            np.random.default_rng(200 + i).standard_normal(16000).astype(np.float32),
            16000,
        )

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "train": [
                    {"speaker_id": "id00001", "wav": f"audio/id00001/{c}"}
                    for c in ("00001.wav", "00002.wav", "00003.wav")
                ]
                + [
                    {"speaker_id": "id00002", "wav": f"audio/id00002/{c}"}
                    for c in ("00001.wav", "00002.wav")
                ],
                "val": [],
            }
        )
    )
    ds = VoxCeleb2MixDataset(tmp_path, split="train", clip_seconds=1.0, num_items=20, seed=0)
    for idx in range(len(ds)):
        s = ds[idx]
        # Enrollment must NOT equal target (different clip => different bytes).
        assert s["enrollment"].shape == s["target"].shape
        if not torch.allclose(s["enrollment"], s["target"], atol=1e-6):
            return  # found at least one idx where enrollment != target
    pytest.fail("every sample had enrollment == target; the 'different clip' logic is broken")
