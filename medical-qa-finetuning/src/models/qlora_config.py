# LoRA configuration
from peft import LoraConfig, TaskType

def get_qlora_config():
    return LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        modules_to_save=["embed_tokens", "lm_head"]
    )
