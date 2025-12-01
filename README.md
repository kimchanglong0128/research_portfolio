# Changlong Jin — Generative Modeling & Multimodal Learning

Hi, I'm **Changlong Jin**, a Master–Ph.D. student at **Yonsei University**.  
My research focuses on **inductive biases in generative models**, especially diffusion-based
identity consistency, multimodal alignment, and parameter-efficient adaptation.

I aim to build **controllable, interpretable, and self-improving generative systems** by
analyzing their underlying dynamics and structural constraints.

---

# Research Focus

- **Inductive Biases in Generative Models**
- **Identity-Stable Diffusion SDE/ODE Dynamics**
- **Closed-Loop T2I ↔ I2T Multimodal Alignment**
- **Parameter-Efficient Adaptation (LoRA, Adapters)**
- **Continuous/Lifelong Learning for Generative Models**
- **Multimodal Fusion (Audio × Image × Text)**

---

# Current Research Direction

## **Proposal 1 — Identity-Preserving Diffusion Dynamics**
Identity features often drift in one-shot personalization.  
I study how identity information propagates through **diffusion SDE/ODE fields**, and design:

- low-rank/adaptive structure as identity-preserving inductive biases  
- identity-stable probability-flow ODE  
- dynamics-based controls for identity consistency under minimal samples  

**Goal:** A theoretically grounded, identity-stable personalization mechanism.

---

## **Proposal 2 — Closed-Loop Multimodal Alignment**
Current T2I models lack self-correction.  
I propose a **T2I ↔ I2T cycle** with:

- CLIP-based alignment rewards  
- cycle consistency to reduce semantic drift  
- replay + adapter-based continuous refinement  

**Goal:** A self-improving multimodal generative system.

