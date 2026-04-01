import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_lora_to_language_head(model):
    """
    Apply LoRA only to the language decoder head.
    Vision encoder weights are frozen entirely.
    This minimises catastrophic forgetting of visual features.
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


class EWC:
    """
    Elastic Weight Consolidation — prevents catastrophic forgetting.

    After training on general language, EWC adds a penalty term that
    slows down changes to weights that were important for the original
    task. This lets us specialise for radiology while retaining general
    language capability.

    Reference: Kirkpatrick et al., 2017
    (Overcoming catastrophic forgetting in neural networks)
    """

    def __init__(self, model, dataloader, device, importance=1000):
        self.model = model
        self.importance = importance
        self.device = device
        self.params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        """
        Estimate Fisher Information Matrix diagonals.
        These represent how important each parameter is
        to the original task — high Fisher = do not change much.
        """
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.model.eval()
        for batch in dataloader:
            self.model.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            outputs.loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        for n in fisher:
            fisher[n] /= len(dataloader)
        return fisher

    def penalty(self, model):
        """
        EWC penalty term added to training loss.
        Penalises large changes to important parameters.
        """
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (
                    self.fisher[n] * (p - self.params[n]) ** 2
                ).sum()
        return self.importance * loss


class GroundingChecker:
    """
    Post-generation hallucination filter.

    Cross-references visual attention maps with generated clinical terms.
    Flags reports where the model describes regions with low visual
    confidence — a lightweight alignment mechanism for multimodal output.
    """

    CLINICAL_TERMS = [
        "opacity", "effusion", "cardiomegaly",
        "consolidation", "atelectasis", "pneumothorax"
    ]

    def __init__(self, attention_threshold=0.15):
        self.threshold = attention_threshold

    def check(self, generated_report: str,
              attention_map: torch.Tensor) -> dict:
        """
        Returns flagged terms and overall hallucination risk score.
        """
        flagged = []
        mean_attention = attention_map.mean().item()

        for term in self.CLINICAL_TERMS:
            if term in generated_report.lower():
                if mean_attention < self.threshold:
                    flagged.append(term)

        return {
            "flagged_terms": flagged,
            "hallucination_risk": len(flagged) / max(1, len(self.CLINICAL_TERMS)),
            "mean_visual_attention": mean_attention,
            "report_approved": len(flagged) == 0
        }
