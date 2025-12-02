# Identity Dynamics — Proposal 1

This module contains experiments for **Identity-Preserving Diffusion Dynamics via Structure-Aware Inductive Biases**.

The main goal is to:
- Analyze identity drift along diffusion SDE/ODE trajectories.
- Evaluate how different parameter-efficient methods (LoRA, adapters, etc.)
  act as inductive biases for identity stability.
- Develop and compare mechanisms for identity-consistent personalization
  under one-shot or few-shot settings.

## Structure

- `lora/` — LoRA-based identity preservation experiments  
- `adapters/` — Alternative adapter-based methods
- `baselines/` — Vanilla fine-tuning / textual inversion, etc.  
- `analysis/` — Notebooks and scripts for visualizing drift, SDE vs ODE