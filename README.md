# 🏫 CHANGLONG JIN — AI Integrated Master–Ph.D. 

I am an Integrated Master–Ph.D. student in Artificial Intelligence at Yonsei University.  
My research focuses on making generative models **more stable, faithful, and self-correcting**, with a particular emphasis on diffusion models and multimodal alignment.

Research Interests:
- **Dynamics-aware Diffusion Models**
- **Closed-loop Multimodal Alignment**
- **PEFT & AST**
- **Inductive Bias**

Overall, my goal is to design generative systems that not only produce high-quality images, but also **know when they are wrong and learn how to fix themselves.**


---

# 🎯 Research Focus


- **Understanding and shaping inductive biases** that gshape identity and semantic representations and influence their stability during diffusion. [**part I**]
- **Modeling personalization as a perturbation to diffusion dynamics**, identifying unstable or drift-prone directions, and designing mechanisms for identity-stable sampling. [**part I**]
- **Building closed-loop multimodal alignment systems** where text–image models use semantic feedback to evaluate and refine their own outputs. [**part II**]




---

# 🔬 Unified Research Agenda  
## Designing Reliable, Dynamics-Aware, and Self-Correcting Generative Models

My research explores how **inductive biases**, **diffusion dynamics**, and **multimodal feedback** shape the reliability of modern generative models.  
I focus on two complementary directions:

1. **Dynamic Control of Drift-Prone Directions in Personalized Diffusion Models**  
   Modeling personalization as a perturbation to the pretrained score field, identifying drift-prone directions, and developing dynamic control mechanisms to preserve identity stability.

2. **Closed-Loop Multimodal Alignment for Text–Image Generation**  
   Enabling text–image generative models to evaluate and refine their outputs using CLIP and caption-based semantic feedback, leading to self-correcting generation pipelines.

Together, these projects aim to establish a unified framework for generative models that maintain identity, preserve semantic consistency, and continually improve through multimodal signals.

---
# 🔵 Part I — Dynamic Control of Drift-Prone Directions in Personalized Diffusion Models


One-shot personalization often leads to **identity drift**, where generated outputs gradually deviate from the target subject.
This issue appears across architectures (U-Net, SDXL, DiT) and tuning methods (LoRA, DreamBooth, Adapters), suggesting that the root cause lies not in data scarcity but in the **instability of diffusion score dynamics and inductive biases**.

---

## 1. Personalization as Score-Field Perturbation

Diffusion models rely on a pretrained score function:

```
s_θ(x_t, t)
```

Personalization modifies it into:

```
s_θ'(x_t, t) = s_θ(x_t, t) + Δs(x_t, t)
```

where `Δs` represents the inductive bias introduced by LoRA or other parameter-efficient tuning methods.

```
s_θ(x_t, t)
```

```
s_θ'(x_t, t) = s_θ(x_t, t) + Δs(x_t, t)
```
Identity features occupy **low-energy, fragile subspaces**, making them highly sensitive to perturbations.

---

## 2. Drift-Prone Directions and Dynamic Instability

Identity drift emerges when personalization perturbations align with unstable directions of the reverse diffusion dynamics:

```
d/dt (δx_t) = J_sθ'(x_t) · δx_t
```

where `J_sθ'` is the Jacobian of the personalized score function.

```
d/dt (δx_t) = J_sθ'(x_t) · δx_t
```


Positive eigenvalues correspond to **drift-prone directions**, where deviations grow over time.
Different diffusion timesteps specialize in structure → texture → identity, so identity instability emerges in specific sampling intervals.


---

## 3. Dynamic Control During Sampling

To counteract drift, a control term is added during sampling:

```
dx_t/dt = s_θ'(x_t, t) + u(x_t, t)
```

```
dx_t/dt = s_θ'(x_t, t) + u(x_t, t)
```

A stabilizing controller suppresses drift-prone score components:

```
u(x_t, t) = -α(t) · P_drift · s_θ'(x_t, t)
```

where `P_drift` projects onto unstable eigen-directions.

```
u(x_t, t) = -α(t) · P_drift · s_θ'(x_t, t)
```


This results in **identity-stable personalization**, independent of architecture or fine-tuning strategy.


---

## 🎯 Goal

**To build a dynamics-aware, theoretically grounded framework that identifies and controls drift-prone directions, enabling stable and reliable one-shot personalization.**



---

# 🔶 Part II — Closed-Loop Multimodal Alignment

Despite high visual quality, diffusion-based text-to-image models often fail to maintain semantic fidelity.  
They lack mechanisms to **evaluate, diagnose, and refine** their own outputs.

I aim to build a closed-loop T2I ↔ I2T refinement system where models learn to self-correct using multimodal feedback.

---

## 1. Multimodal Feedback Modeling

I design a unified reward integrating multiple semantic signals:

- **CLIP similarity** between text and image (global alignment)  
- **caption-based similarity** using an image-to-text model (`ŷ = I2T(x)`) and a text encoder `E(·)`  
- optionally, **perceptual or feature-level consistency** for stability  

A conceptual reward could be written as:

- `R(x, y) = α * CLIP(x, y) + β * cos(E(ŷ), E(y)) + γ * Perceptual(x)`

This creates a richer supervisory signal than CLIP alone and can be used to rank, filter, or refine generations.

---

## 2. Closed-Loop Generation and Refinement

Instead of full RL training, I explore lightweight iterative refinement driven by differentiable feedback.  
Given a current image `x_t`:

- refine in image/latent space using a step like  
  `x_{t+1} = x_t + η * ∇_x R(x_t, y)`  
- or refine the generator parameters with small adapter / LoRA updates guided by `∇_θ R(G_θ(y), y)`  

This enables:

- correction of local attribute mismatches  
- reinforcement of global semantics  
- prevention of drift on long or compositional prompts  

It forms a **self-improving loop** without expensive full-model retraining.  
Reinforcement learning is optional for non-differentiable objectives, but **not required** for the core closed-loop framework.

---

## 3. Representation-Consistent Alignment

Current T2I and I2T systems are often trained separately.  
I aim to enforce **cross-modal consistency** between image and text embeddings:

- encourage `E_img(x)` and `E_text(y)` to be close for aligned pairs  
- encourage `E_text(ŷ)` to be close to `E_text(y)` when `ŷ = I2T(x)`  

A conceptual loss:

- `L_repr = ||E_img(x) - E_text(y)|| + λ * ||E_text(ŷ) - E_text(y)||`

This improves:

- compositional reasoning  
- robustness on long-tail prompts  
- fine-grained attribute consistency  
- semantic grounding across T2I and I2T paths  

**Goal:**  
A T2I model that can understand its own mistakes, and iteratively correct them through multimodal feedback.

---

# 🧭 Preliminary Work

## 📚 Theoretical Preparation

I conducted in-depth reviews and theoretical studies on:

- [Generative & Diffusion Models](https://www.notion.so/Generative-Diffusion-Model-2a4d80fa6bde801fa55bf3e4cdde2e05)  
- [Multimodal Alignment, Representation Learning & Recommendation Systems](https://www.notion.so/1ccd80fa6bde804fbf91cf15ec433298?v=1ccd80fa6bde80b18a97000c46532dd4)  
- Parameter-efficient Fine-Tuning, Selective Adaptive Tuning

(Additional research notes are documented in personal logs.)

---

## 🧪 Experiments & Prototyping

- LoRA-based diffusion personalization experiments  
- Multimodal fusion prototypes (vision × text encoders)  
- Closed-loop caption feedback tests  
- Identity drift visualization tools  
- Diffusion-step identity curve analysis  

---

# 🚀 Research Roadmap

![Roadmap](images/RoadMap.png)

---

# 📫 Contact

- Email: **kimcl1221@yonsei.ac.kr**  
- GitHub: **https://github.com/kimchanglong0128**
