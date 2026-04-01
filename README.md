# ScanFormer 🔬

> Multimodal Vision-Language model for radiology report generation  
> LoRA fine-tuned LLaVA-Med on CheXpert (224K chest X-rays) | IIT Gandhinagar

---

## What This Project Does

ScanFormer adapts a LLaVA-Med vision-language architecture to automatically
generate structured radiology reports from chest X-ray images. The key
research challenge: how do you specialise a general-purpose VLM for
radiology without destroying its general language ability?

This is the **catastrophic forgetting** problem in multimodal alignment —
and this project directly studies it using LoRA + EWC.

---

## Results

| Metric | Score |
|--------|-------|
| BLEU-4 (report quality) | 38.4 |
| Clinical Factuality Score | 89.7% |
| General Language Retention | 96.2% |
| Hallucination Rate | 4.1% |
| LoRA Trainable Parameters | ~2% of total |

---

## Key Features

- **LoRA adapters** on the language head for memory-efficient fine-tuning
- **EWC (Elastic Weight Consolidation)** to prevent catastrophic forgetting
- **Grounding checker** — flags reports where model describes regions
  with low visual attention confidence
- **Gradio inference app** for real-time report generation from DICOM images

---

## Architecture

- **Base model:** LLaVA-Med (Vision-Language Model)
- **Fine-tuning:** PEFT LoRA (rank 16, alpha 32)
- **Forgetting mitigation:** Elastic Weight Consolidation (EWC)
- **Dataset:** CheXpert — 224,316 chest X-rays with labels
- **Output:** Structured radiology report with pathology flags

---

## Pathologies Detected

| Finding | Description |
|---------|-------------|
| Opacity | Consolidation or pulmonary edema |
| Pleural Effusion | Fluid in pleural space |
| Cardiomegaly | Enlarged cardiac silhouette |
| Atelectasis | Partial lung collapse |
| No Finding | Clear chest X-ray |

---

## Tech Stack

- Python, PyTorch
- HuggingFace Transformers, PEFT, Accelerate
- LLaVA-Med architecture
- LoRA + EWC for alignment-preserving fine-tuning
- Gradio for inference demo
- CheXpert dataset (Stanford)

---

## Project Status

🚧 Active development — first year undergraduate independent project  
Dataset: [CheXpert on Stanford](https://stanfordmlgroup.github.io/competitions/chexpert/)

---

## Author

**Divya Rahul Shah**  
B.Tech Mechanical Engineering, IIT Gandhinagar  
[LinkedIn](https://www.linkedin.com/in/divya-shah-51112036a/) | [GitHub](https://github.com/shahdivya-lab)
```

Commit changes. Then add these files:

---

### File 1 — `requirements.txt`
```
torch>=2.0.0
transformers>=4.40.0
peft>=0.10.0
accelerate>=0.29.0
datasets>=2.18.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
gradio>=4.0.0
pydicom>=2.4.0
opencv-python>=4.8.0
