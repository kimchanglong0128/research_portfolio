# 🏫 CHANGLONG JIN — Integrated Master–Ph.D. Student, Yonsei University AI

Hi, I'm **CHANGLONG JIN**, an Integrated Master–Ph.D. student in Artificial Intelligence at **Yonsei University**.  
My research centers on building **faithful and self-improving generative models** by designing stronger inductive biases, stabilizing diffusion dynamics, and leveraging multimodal feedback for continuous refinement.

I aim to bridge the gap between modern generative models’ impressive visual quality and their lack of **identity consistency**, **semantic faithfulness**, and **self-correction capabilities**.

My interests include:
- score-based generative modeling  
- multimodal alignment & consistency  
- dynamics of diffusion models  
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

My research agenda consists of **two complementary components** that together aim to enhance the reliability, controllability, and semantic stability of diffusion-based generative systems.

---

## 🔷 **Part I — Identity-Preserving Diffusion Dynamics**

Generative models often fail to maintain identity fidelity in one-shot personalization.  
To address this, I study how identity-related information propagates through **diffusion SDE/ODE fields** and design mechanisms that stabilize identity across the generative trajectory.

I focus on:
- analyzing identity evolution through intermediate diffusion states  
- understanding how low-rank adaptation reshapes identity-sensitive directions  
- designing identity-corrective vector fields and identity-stable probability-flow ODEs  

**Goal:** Build diffusion models with intrinsic inductive biases that preserve identity under minimal-sample personalization.

---

## 🔶 **Part II — Closed-Loop Multimodal Alignment**

Text-to-image models frequently produce misaligned images lacking semantic fidelity.  
I propose a **closed-loop T2I ↔ I2T refinement framework** where generative models use multimodal feedback to continuously improve alignment.

Key ideas include:
- CLIP- and caption-based multimodal alignment rewards  
- iterative self-correction through gradient-based refinement  
- representation-consistent training across T2I and I2T pathways  
- lightweight continuous updates using adapters or LoRA  

**Goal:** Enable generative models to self-evaluate, self-correct, and maintain robust semantic alignment—even for long-tail or compositional prompts.

---

# 🧭 Preliminary Work

### 📚 Theoretical Preparation
Extensive study and synthesis of:
- Generative and diffusion model theory  
- Multimodal alignment & representation learning  
- Parameter-efficient fine-tuning  
- Score-based model interpretation  

(Research notes maintained in personal logs.)

---

### 🧪 Experiments & Prototyping
- LoRA-based diffusion personalization experiments  
- Cycle-consistency analysis across T2I ↔ I2T  
- Multimodal fusion prototypes (vision × text encoders)  
- Diffusion trajectory visualization and identity drift measurement  

---

# 🚀 Research Roadmap
![Roadmap](images/RoadMap.png)

---

# 📫 Contact

- Email: **kimcl1221@yonsei.ac.kr**  
- GitHub: **https://github.com/kimchanglong0128**
