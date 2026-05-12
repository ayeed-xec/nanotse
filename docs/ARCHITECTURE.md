# NanoTSE — Architecture

> Single source of truth for module boundaries, shapes, streaming contracts,
> and implementation order. Update this **before** writing a new module —
> code without a row in this file is over-engineering.

## What it does, end to end

> Point a camera at a few people, put a mic in a noisy room, and NanoTSE
> gives you the clean voice of the active person who is seen and talking,
> and remembers them by voice and face across the streaming session.

The model auto-selects the active speaker (no user "select" step). Slot
memory is per-session, fixed capacity, LRU-evicted. Three paper contributions:
named-slot face+voice memory, streaming AV backbone, IBA cross-session metric.

## Constants

| Symbol | Value | Where it shows up |
|---|---|---|
| `sample_rate` | 16 000 Hz | audio everywhere |
| `fps` | 25 | visual stream rate |
| audio frame rate | 100 Hz | after audio frontend (hop = 160 samples) |
| `chunk_ms` | 40 ms | streaming chunk (= 4 audio frames, = 1 video frame) |
| target latency | < 60 ms | p95 chunk-latency budget |
| `N` (slots) | **16** default; ablate over `{1, 4, 16, 32}` | slot memory capacity |
| `D` (audio feat dim) | **256** | audio frontend output |
| `Dv` (visual feat dim) | **512** | visual frontend output |
| `S` (slot dim) | **256** | face_emb dim = voice_emb dim |

## Data flow

```
                                INPUT
       ┌──────────────────────────────────────────────────────┐
       │  audio: (B, T)              float32  @ 16 kHz         │
       │  video: (B, F, H, W, 3)     uint8    @ 25 fps         │
       └──────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┴────────────────────────┐
        ▼                                                  ▼
┌──────────────────────┐                       ┌──────────────────────┐
│ AudioFrontend        │                       │ VisualFrontend       │
│ Conv1D (+ STFT)      │                       │ mouth-ROI / AV-HuBERT│
│ (B, T) → (B, Ta, D)  │                       │ (B,F,H,W,3) →        │
│ Ta @ 100 Hz          │                       │   (B, Tv, Dv)        │
└──────────┬───────────┘                       │ Tv @ 25 Hz, frozen   │
           │                                   └──────────┬───────────┘
           │                                              │
           ▼                                              ▼
        ┌─────────────────────────────────────────────────────┐
        │ DualCacheFusion                                     │
        │ visual KV cache (Tv, Dv)  @ 25 Hz                   │
        │ audio queries    (Ta, D)   @ 100 Hz                 │
        │ → fused (B, Ta, D′)                                 │
        └────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
                ┌────────────────────────────────────────┐
                │ NamedSlotMemory                        │
                │ N slots × (face_vec, voice_vec)        │
                │ Locatello slot competition (softmax    │
                │ over slots), GRU-EMA update,           │
                │ LRU eviction when full                 │
                │ → (B, Ta, D′ + S), slots (B, N, 2S)    │
                └────────────────────┬───────────────────┘
                                     │
                                     ▼
                  ┌──────────────────────────────────┐
                  │ Backbone                         │
                  │ CUDA:     Mamba2Backbone         │
                  │ MPS/edge: ChunkAttnBackbone      │
                  │ → (B, Ta, D″)                    │
                  └──────────────────┬───────────────┘
                                     │
            ┌────────────────────────┴────────────────────────┐
            ▼                                                 ▼
    ┌──────────────────┐                          ┌──────────────────┐
    │ TSEHead          │                          │ ASDHead          │
    │ mask + iSTFT     │                          │ per-slot logits  │
    │ → (B, T)         │                          │ → (B, Ta, N)     │
    └────────┬─────────┘                          └────────┬─────────┘
             │                                             │
             ▼                                             ▼
       clean voice                                  active-speaker pick
       of active speaker                            (which slot is talking)
```

## Modules — one row per file, no orphans

| Path | Role | I/O | Streaming state |
|---|---|---|---|
| [nanotse/models/frontends/audio_stft.py](../nanotse/models/frontends/) `AudioFrontend` | Conv1D encoder; STFT branch optional in W2.5 | `(B,T)` → `(B,Ta,D)` | none |
| [nanotse/models/frontends/visual_avhubert.py](../nanotse/models/frontends/) `VisualFrontend` | mouth-ROI CNN (W3.1) or frozen AV-HuBERT (W3.2) | `(B,F,H,W,3)` → `(B,Tv,Dv)` | none |
| [nanotse/models/fusion/dual_cache.py](../nanotse/models/fusion/) `DualCacheFusion` | cross-modal attn, visual KV @ 25 Hz / audio Q @ 100 Hz | `(audio, visual)` → `(B,Ta,D′)` | rolling visual KV cache |
| [nanotse/models/memory/slot_attention.py](../nanotse/models/memory/) `NamedSlotMemory` | Locatello slot competition + GRU-EMA + LRU | `(B,Ta,D′)` → `(B,Ta,D′+S)`, slots `(B,N,2S)` | slot bank + LRU timestamps |
| [nanotse/models/backbones/chunk_attn.py](../nanotse/models/backbones/) `ChunkAttnBackbone` | causal chunked attention (MPS/edge) | `(B,Ta,X)` → `(B,Ta,X)` | rolling KV cache |
| [nanotse/models/backbones/mamba2.py](../nanotse/models/backbones/) `Mamba2Backbone` | Mamba-2 SSM (CUDA only, W4) | `(B,Ta,X)` → `(B,Ta,X)` | SSM state |
| [nanotse/models/heads/tse.py](../nanotse/models/heads/) `TSEHead` | mask prediction + iSTFT | `(B,Ta,X)` → `(B,T)` | none |
| [nanotse/models/heads/asd.py](../nanotse/models/heads/) `ASDHead` | per-slot active-speaker logits | `(B,Ta,X), slots` → `(B,Ta,N)` | none |
| [nanotse/models/nanotse.py](../nanotse/models/) `NanoTSE` | top-level assembly | `(audio, video)` → `(clean_audio, asd_logits)` | composed |
| [nanotse/models/baselines/tdse.py](../nanotse/models/baselines/tdse.py) `TDSEBaseline` | audio-only Conv-TasNet-lite (already landed) | `(B,T)` → `(B,T)` | none |

Each row matches **one** paper contribution component or **one** required
input/output stage. No "framework" layer, no abstract base classes, no
plugin registry. If a file would be added that isn't in this table, it
shouldn't be added.

## Streaming contract

Every module whose name ends in `…Cache`, `…Backbone`, or `…Memory` conforms to:

```python
class StreamingModule(nn.Module):
    def init_state(self, batch_size: int, device: torch.device) -> StateT: ...
    def forward_chunk(self, x: Tensor, state: StateT) -> tuple[Tensor, StateT]: ...
    def forward(self, x: Tensor) -> Tensor:
        # Offline / training path: process full sequence in one call.
        ...
```

`StateT` is a per-module concrete dataclass — **no globals**, **no module
attributes that mutate during forward()**. State is always passed in/out
explicitly so the loop in `scripts/train.py` and the latency bench own it.

Frontends and heads are stateless (no `init_state` / `forward_chunk`).

## Losses — multi-task, added in order

| Loss | What it supervises | When added |
|---|---|---|
| `L_tse  = negative_si_snr(est, target)` | TSE head extraction quality | **W2.4** (TDSE baseline → full NanoTSE) |
| `L_infonce(slot_emb, speaker_id)` | slot embeddings pull same-speaker close, push others apart | **W3.3** (with NamedSlotMemory) |
| `L_asd  = BCE(asd_logits, gt_speaker_one_hot)` | ASD head selects the currently talking slot | **W3.4** |
| `L_consistency = MSE(slot_emb @ t, slot_emb @ t+gap)` | slot stability under temporary speaker exit | **W3.5** |
| `L = α·L_tse + β·L_asd + γ·L_infonce + δ·L_consistency` | combined | **W4** (first A100 burst) |

Defaults: `α=1, β=0.2, γ=0.1, δ=0.1`. Tune in W5 ablations.

PCGrad (gradient surgery from av-listen) added **only if** training shows
loss conflicts (monitor with `Tracker` — gradient cosine between losses).
Don't add it speculatively.

## Implementation order

| Sprint | Lands | Gate |
|---|---|---|
| **W2.1** | `AudioFrontend` (Conv1D only) | shape contract test |
| **W2.2** | `ChunkAttnBackbone` (causal, KV cache) | `forward_chunk` ≡ `forward` on one chunk |
| **W2.3** | `TSEHead` (mask + iSTFT round-trip) | iSTFT(STFT(x)) ≈ x |
| **W2.4** | Wire as audio-only `NanoTSE` (no visual yet) | M3 8-clip overfit ≥ +10 dB SI-SDRi on **real** speech |
| **W2.5** | Real `VoxCeleb2MixDataset` (depends on user running fetch script) | speaker-disjoint train/val split regression test |
| **W3.1** | `VisualFrontend` (mouth-ROI CNN; AV-HuBERT load comes later) | random video → embedding shape |
| **W3.2** | `DualCacheFusion` | cache equivalence test |
| **W3.3** | `NamedSlotMemory` | 2-speaker bootstrap test; speaker re-entry rebinds same slot |
| **W3.4** | `ASDHead` | per-slot softmax logits |
| **W3.5** | Full `NanoTSE.forward()` (all heads, slot memory live) | smoke train: NanoTSE ≥ MeMo+0.5 dB on M3 |
| **W4**  | `Mamba2Backbone` + first A100 burst | reproduce MeMo ≥ 9.85 dB SI-SNR |

## Test conventions

- **Every module:** shape contract (input shape → output shape).
- **Every streaming module:** `forward_chunk` chained over T chunks ≡ `forward` on the concatenated input. Within float-32 tolerance.
- **`NamedSlotMemory`:** 2-speaker deterministic test:
  - speaker A talks alone → slot[A] populated, slot[B] empty;
  - speaker B talks alone → slot[B] populated, slot[A] unchanged;
  - speaker A returns → same slot[A] rebinds (cosine-sim ≥ 0.9), no new slot allocated.
- **Full model:** 8-clip overfit on synthetic + real (once data fetched). Real-data bar is the W2 paper gate (+10 dB SI-SDRi).
- **Latency:** `nanotse/eval/latency_bench.py` measures p50/p95/p99 per-chunk. p95 must stay under 60 ms on M3 Pro and an i7 CPU.

## What is NOT in scope (do not build)

- **Cross-session persistent identity.** Slots wipe at session end. Persistence across reboot = product extension, not paper.
- **Multi-language support.** VoxCeleb2-mix is the dataset; whatever languages it contains, contains. No special handling.
- **Far-field eval / 3D-Speaker pivot.** Per prior discussion, kills VoxCeleb2 comparability and the June deadline.
- **Privacy-preserving embeddings, federated learning, on-device encryption.** Out of paper scope.
- **Real-time microphone input pipeline / audio device drivers.** The paper trains and evaluates on chunked file input; no production audio I/O.
- **TTS / ASR / translation downstream stages.** NanoTSE outputs clean audio; downstream apps are someone else's stack.
- **A "framework" layer** (model registry, plugin system, generic streaming runtime). Direct imports, explicit assembly in `nanotse/models/nanotse.py`. If three modules need the same pattern, refactor *then*, not now.

## Why this and not something fancier

Each module is one of: a paper contribution component, a required I/O stage, or a planned baseline. There is no module whose only justification is "we might need this for X someday." The streaming contract is the only abstraction, and it exists because three independent modules (`DualCacheFusion`, `NamedSlotMemory`, `ChunkAttnBackbone`) genuinely share the same shape — that's the threshold for abstracting.

When in doubt: do not add. The W7–W8 paper-writing phase has enough work without retrofitting around speculative architecture choices.
