
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login
 

# Check M3 compatibility
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load Student Model & Tokenizer (Quantized)
student_model_name = "distilgpt2"  
student_model = AutoModelForCausalLM.from_pretrained(student_model_name, torch_dtype=torch.bfloat16)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, padding_side="left")
student_tokenizer.pad_token = student_tokenizer.eos_token


# Load Teacher Model (Minimal Memory Use)
teacher_model_name = "EleutherAI/gpt-neo-1.3B"  
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu"  
)
teacher_model.eval()

# Distillation Loss
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    return alpha * kl_loss + (1 - alpha) * ce_loss

class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch= None):
        # Move inputs to student device (mps)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Teacher inference on CPU, move logits to mps
        with torch.no_grad():
            teacher_outputs = teacher_model(**{k: v.to("cpu") for k, v in inputs.items()})
            teacher_logits = teacher_outputs.logits.to(device)
        
        # Student forward
        student_outputs = model(**inputs)
        
        # Compute loss
        loss = distillation_loss(student_outputs.logits, teacher_logits, inputs["labels"])
        return (loss, student_outputs) if return_outputs else loss

# Dataset (Smaller Subset, Shorter Sequences)
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:1%]")  # Subset for testing
def tokenize_function(examples):
    tokenized = student_tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=128,  # Reduced from 512
        padding="max_length"
    )
    labels = tokenized["input_ids"].copy()
    labels = [[label if label != student_tokenizer.pad_token_id else -100 for label in seq] for seq in labels]
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

# Training Arguments (Optimized for M3)
training_args = TrainingArguments(
    output_dir="./distill_output",
    per_device_train_batch_size=1,  # Tiny batch size
    gradient_accumulation_steps=16,  # Accumulate over more steps
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=5,
    save_strategy="no",
    dataloader_num_workers=0,
    run_name="DeepSeek_Distillation_Run_1",

)

# Trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=student_tokenizer,
)

# Train
trainer.train()

# Save
student_model.save_pretrained("./distilled_student_model")
student_tokenizer.save_pretrained("./distilled_student_model")