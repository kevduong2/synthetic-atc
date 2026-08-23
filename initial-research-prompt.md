Synthetic ATC Audio Data Generation — Research & Design Task
Context

We're building a synthetic audio data pipeline to train an ASR model for ATC (air traffic control) radio transcription. We have small collected live datasets and want to expand them with synthetic audio. Another team owns the labeled text/scenario data — our scope is audio generation only: taking text and producing realistic, noisy ATC radio audio (though we should support voice-level variation like speed, pitch, and accent changes).

TTS is handled by the Kokoro model; treat it as a fixed dependency.

Objectives

Build a generator with two configurable modes:

Mode 1 — Procedural (fully independent baseline).
A procedurally generated radio-noise pipeline layering common radio artifacts (static, squelch, clipping, band-limiting, interference, dropouts, etc.) over TTS output. Randomized and highly varied so models trained on it generalize to unknown ATC audio conditions. No dependency on collected data.

Mode 2 — Calibrated (learned from real samples).
A generative approach (GAN, diffusion, or whatever current research supports best — evaluate options) that learns the noise/channel characteristics of a small real dataset and generates synthetic audio matching it. Target use case: expand a real dataset of ~1,000 samples to ~10,000 (9k synthetic), deliberately oversampling underrepresented cases like critical/emergency transmissions and rare vocabulary.

Phases

Phase 1 — Research. Survey the 2026 state of the art in: synthetic speech data augmentation for ASR, radio/channel noise simulation, generative audio models for domain adaptation (GAN vs. diffusion vs. alternatives), and any ATC-specific ASR or dataset work. Cite sources.

Phase 2 — Codebase analysis. Review this codebase and map how the research findings apply: what exists, what's reusable, what's missing, and where each mode's components would live.

Phase 3 — Design & planning docs. Produce thorough design and implementation plan documents suitable for handoff to coding agents. Split into multiple plan files if that improves clarity (e.g., one per mode, plus a shared architecture doc). Include architecture, component breakdowns, data flow, config schema for the two modes, evaluation strategy for synthetic data quality, and a phased implementation roadmap with clear task boundaries.

Constraints
Do not write code. Deliverables are research findings and plan/design documents only.
Don't design text/scenario generation — that's the other team's scope.
Both modes should share as much infrastructure as practical (augmentation primitives, config system, output format).


Source to our demo audio clips(sample a few of these and copy them into our current repo. sample like 100 for us to calibrate off of)
/Users/kevin/repos/ai/asr-demo/clips/library/OLA-0645d24d
/Users/kevin/repos/ai/asr-demo/clips/library/OLA-0645d249
