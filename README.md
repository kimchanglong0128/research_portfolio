# 🏫 CHANGLONG JIN — Integrated Master–Ph.D. Student, Yonsei University AI

Hi, I'm **CHANGLONG JIN**, an Integrated Master–Ph.D. student in Artificial Intelligence at **Yonsei University**.  
My research centers on building **faithful and self-improving generative models** by designing stronger inductive biases, stabilizing diffusion dynamics, and leveraging multimodal feedback for continuous refinement.

I aim to bridge the gap between modern generative models’ impressive visual quality and their lack of **identity consistency**, **semantic faithfulness**, and **self-correction capabilities**.

Research interests include:
- score-based generative modeling  
- multimodal alignment & consistency  
- diffusion dynamics  
- self-refining generative systems  
- parameter-efficient adaptation (LoRA, adapters)  


---

# 🎯 Research Focus

- **Inductive Biases in Generative Models**  
- **Identity-Stable Diffusion SDE/ODE Dynamics**  
- **Closed-Loop Text–Image ↔ Image–Text Multimodal Alignment**  
- **Parameter-Efficient Personalization & Adaptation**  
- **Self-Correction and Feedback-Driven Refinement**  
- **Multimodal Fusion (Image × Text)**  
- **Improving Semantic Faithfulness in T2I Models**  


---

# 🔬 Unified Research Agenda  
## **Toward Faithful & Self-Improving Generative Models**

This agenda contains **two complementary research directions** that together aim to enhance the consistency, controllability, and semantic reliability of diffusion-based generative models.

---

# 🔷 Part I — Identity-Preserving Diffusion Dynamics

One-shot personalization frequently suffers from *identity drift*, where the generated subject gradually deviates from the reference.  
This issue persists across architectures—including U-Net, SDXL, and DiT—suggesting that identity loss originates from **diffusion dynamics**, not model capacity.

My research investigates *how identity information propagates through the diffusion SDE/ODE process* and how to stabilize it.

---

## **1. Identity Flow Analysis Through Diffusion Timesteps**

I decode intermediate diffusion states (latents \(x_t\)) and compute identity embeddings using CLIP/DINO:

- visualize identity similarity curves across timesteps  
- identify critical intervals where identity degradation emerges  
- compare DDPM / DDIM / Flow-Matching behaviors  
- analyze how noise schedule and timestep parameterization affect identity sensitivity  

This produces the first systematic **identity evolution profile** for diffusion models.

---

## **2. Effects of Low-Rank Personalization on Identity Stability**

Low-rank adaptation (LoRA) modifies the score function:

- Which layers help identity retention?  
- Which layers induce drift?  
- Does rank or scale correlate with stability?  
- Do LoRA update directions align with identity-preserving subspaces?  

Using gradient projections and embedding-space perturbation analysis, I study the **structure–identity relationship** inside diffusion models.

---

## **3. Designing Identity-Stable Diffusion Dynamics**

The goal is to build diffusion mechanisms that *naturally* preserve identity:

- identity-corrective vector fields added to the probability-flow ODE  
- projection onto identity-consistent subspaces of the score field  
- sampling-time regularization to stabilize sensitive regions  
- architecture-agnostic controls applicable to U-Net, SDXL, DiT  

**Goal:**  
A theoretically grounded framework for *identity-stable personalization*, even in one-shot settings.

---

# 🔶 Part II — Closed-Loop Multimodal Alignment

Despite high generative quality, diffusion models often fail to maintain strong semantic fidelity.  
They lack mechanisms to **evaluate, diagnose, and refine** their own outputs.

I aim to build a closed-loop T2I ↔ I2T refinement system where models learn to self-correct using multimodal feedback.

---

## **1. Multimodal Feedback Modeling**

I design a unified reward integrating three semantic signals:

- **CLIP similarity** (global alignment)  
- **caption-based similarity** (fine-grained semantics)  
- **perceptual consistency** (image-level stability)  

\[
R(x, y) = 
\alpha \cdot \mathrm{CLIP}(x,y)
+ \beta \cdot \cos(E(\hat{y}), E(y))
+ \gamma \cdot \mathrm{Perceptual}(x)
\]

This creates a richer supervisory signal than CLIP alone.

---

## **2. Closed-Loop Generation and Refinement**

Instead of full RL-based training (high cost), I explore lightweight iterative refinement:

\[
x_{t+1} = x_t + \eta \nabla_x R(x_t, y)
\]

This mechanism enables:

- correction of local attribute mismatches  
- reinforcement of global semantics  
- prevention of drift during long or compositional prompts  

It forms a **self-improving loop** without expensive retraining.

---

## **3. Representation-Consistent Alignment**

I enforce coherence across T2I and I2T embedding spaces:

\[
\mathcal{L}_{repr} =
\|E_{\mathrm{img}}(x) - E_{\mathrm{text}}(y)\|
+ \lambda \|E(\hat{y}) - E(y)\|.
\]

This improves:

- compositional reasoning  
- long-tail robustness  
- fine-grained attribute consistency  
- cross-modal semantic grounding  

**Goal:**  
A T2I model that understands—and corrects—its own mistakes.

---

# 🧭 Preliminary Work

### 📚 Theoretical Preparation
Extensive study on:
- generative & diffusion model theory  
- multimodal alignment and representation learning  
- parameter-efficient tuning  
- diffusion trajectory interpretation  

(Notes maintained in private logs.)

---

### 🧪 Experiments & Prototyping
- LoRA-based personalization experiments  
- multimodal fusion prototypes  
- closed-loop caption feedback tests  
- identity drift visualization tools  
- diffusion-step identity curve analysis  

---

# 🚀 Research Roadmap
![Roadmap](images/RoadMap.png)

---

# 📫 Contact

- Email: **kimcl1221@yonsei.ac.kr**  
- GitHub: **https://github.com/kimchanglong0128**
