import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models and tokenizers
teacher_model_name = "EleutherAI/gpt-neo-1.3B"
distilled_model_path = "./distilled_student_model"

# Teacher model with bfloat16 to save memory
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name, torch_dtype=torch.bfloat16
).to(device)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# Distilled model (assuming same tokenizer unless specified otherwise)
distilled_model = AutoModelForCausalLM.from_pretrained(distilled_model_path).to(device)
distilled_tokenizer = AutoTokenizer.from_pretrained(distilled_model_path)

# Define task configurations with label_field
task_configs = {
    "SuperGLUE_cb": {
        "prompt_template": "Premise: {premise} Hypothesis: {hypothesis} The relationship is",
        "label_map": {0: "entailment", 1: "contradiction", 2: "neutral"},
        "input_fields": {"premise": "premise", "hypothesis": "hypothesis"},
        "label_field": "label"
    },
    "GLUE_mrpc": {
        "prompt_template": "Sentence1: {sentence1} Sentence2: {sentence2} Are they equivalent or not equivalent? The answer is",
        "label_map": {0: "not equivalent", 1: "equivalent"},
        "input_fields": {"sentence1": "sentence1", "sentence2": "sentence2"},
        "label_field": "label"
    },
    "XTREME_XNLI": {
        "prompt_template": "Premise: {premise} Hypothesis: {hypothesis} The relationship is",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "input_fields": {"premise": "sentence1", "hypothesis": "sentence2"},
        "label_field": "gold_label"
    },
    "TREC": {
        "prompt_template": "Classify this question: {text} into one of: abbreviation, entity, description, human, location, numeric. The category is",
        "label_map": {0: "abbreviation", 1: "entity", 2: "description", 3: "human", 4: "location", 5: "numeric"},
        "input_fields": {"text": "text"},
        "label_field": "coarse_label"
    }
}

# Define benchmarks (classification tasks only)
benchmarks = [
    {"name": "SuperGLUE_cb", "dataset": "super_glue", "config": "cb", "split": "validation"},
    {"name": "GLUE_mrpc", "dataset": "glue", "config": "mrpc", "split": "validation"},
    {"name": "XTREME_XNLI", "dataset": "xtreme", "config": "XNLI", "split": "test"},
    {"name": "TREC", "dataset": "trec", "config": None, "split": "test"}
]

# Evaluation function
def evaluate_model(model, tokenizer, benchmark, task_config):
    """Evaluate a model on a benchmark task using zero-shot classification."""
    # Load dataset
    if benchmark["config"]:
        dataset = load_dataset(benchmark["dataset"], benchmark["config"], split=benchmark["split"])
    else:
        dataset = load_dataset(benchmark["dataset"], split=benchmark["split"])

    if benchmark["name"] == "XTREME_XNLI":
        dataset = dataset.filter(lambda x: x["language"] == "en")

    correct = 0
    total = 0
    for example in dataset:
        # Construct prompt
        input_data = {prompt_var: example[dataset_field] for prompt_var, dataset_field in task_config["input_fields"].items()}
        prompt = task_config["prompt_template"].format(**input_data)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Get true label using the specified label_field
        true_label = example[task_config["label_field"]]

        # Convert string labels to integers for XTREME_XNLI
        if benchmark["name"] == "XTREME_XNLI":
            label_str_to_int = {"entailment": 0, "neutral": 1, "contradiction": 2}
            true_label = label_str_to_int[true_label]

        # Compute loss for each verbalizer
        verbalizer_losses = []
        for label_idx, verbalizer in task_config["label_map"].items():
            verbalizer_ids = tokenizer.encode(verbalizer, add_special_tokens=False)
            input_ids = prompt_ids + verbalizer_ids
            input_ids_tensor = torch.tensor([input_ids]).to(device)
            labels = [-100] * len(prompt_ids) + verbalizer_ids
            labels_tensor = torch.tensor([labels]).to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids_tensor, labels=labels_tensor)
                loss = outputs.loss.item()
            verbalizer_losses.append(loss)

        # Predict label with smallest loss
        pred_idx = np.argmin(verbalizer_losses)
        pred_label = list(task_config["label_map"].keys())[pred_idx]
        if pred_label == true_label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

# Collect results
results_data = []

# Evaluate both models on each benchmark
for benchmark in benchmarks:
    task_name = benchmark["name"]
    if task_name in task_configs:
        task_config = task_configs[task_name]
        print(f"Evaluating {task_name}...")
        
        # Teacher model evaluation
        teacher_accuracy = evaluate_model(teacher_model, teacher_tokenizer, benchmark, task_config)
        
        # Distilled model evaluation
        distilled_accuracy = evaluate_model(distilled_model, distilled_tokenizer, benchmark, task_config)
        
        # Store results
        results_data.append({
            "Workload": task_name,
            "Original Model Accuracy": teacher_accuracy,
            "Distilled Model Accuracy": distilled_accuracy,
            "Remarks": "Zero-shot classification performance"
        })
        print(f"{task_name} - Teacher Accuracy: {teacher_accuracy:.4f}, Distilled Accuracy: {distilled_accuracy:.4f}")

# Save results to CSV
df = pd.DataFrame(results_data)
df.to_csv("benchmark_results.csv", index=False)
print("Evaluation completed. Results saved to 'benchmark_results.csv'")