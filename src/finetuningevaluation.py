import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models and tokenizers
teacher_model_name = "EleutherAI/gpt-neo-1.3B"
distilled_model_path = "./yahoo_finance_finetuned"

# Teacher model with bfloat16 to save memory
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name, torch_dtype=torch.bfloat16
).to(device)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# Distilled model (assuming same tokenizer unless specified otherwise)
distilled_model = AutoModelForCausalLM.from_pretrained(distilled_model_path).to(device)
distilled_tokenizer = AutoTokenizer.from_pretrained(distilled_model_path)


task_config_finqa = {
    "prompt_template": "Question: {question} Context: {context} The answer is",
    "label_map": {},
    "input_fields": {"question": "question", "context": "context"},
    "label_field": "answer"
}

# Define benchmark for FinQA
benchmark_finqa = {
    "name": "TheFinAI/Fino1_Reasoning_Path_FinQA",
    "dataset": "TheFinAI/Fino1_Reasoning_Path_FinQA",
    "config": None,
    "split": "train" 
}

# Function to compute token-level F1 score
def compute_token_f1(true_str, pred_str):
    true_tokens = true_str.strip().split()
    pred_tokens = pred_str.strip().split()
    common_tokens = set(true_tokens) & set(pred_tokens)
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return int(pred_tokens == true_tokens)
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(true_tokens)
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Function to evaluate the model on the FinQA task
def evaluate_qa_model(model, tokenizer, benchmark, task_config):
    """Evaluate a question-answering model on the FinQA dataset."""
    # Load dataset
    dataset = load_dataset(benchmark["dataset"], split=benchmark["split"])

    correct = 0
    total = 0
    f1_scores = []

    for example in dataset:
        # Construct prompt
        input_data = {prompt_var: example[dataset_field] for prompt_var, dataset_field in task_config["input_fields"].items()}
        prompt = task_config["prompt_template"].format(**input_data)

        # Tokenize the prompt with special tokens (include EOS, BOS, etc.)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        input_ids = inputs.input_ids

        # Generate model output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)

        # Slice generated output to exclude input prompt tokens
        generated_ids = outputs[0][input_ids.shape[1]:]  # Exclude input prompt tokens
        predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # True answer
        true_answer = example[task_config["label_field"]]

        # Compute Exact Match (EM) and token-level F1 score
        if predicted_answer.strip() == true_answer.strip():
            correct += 1

        f1 = compute_token_f1(true_answer.strip(), predicted_answer.strip())
        f1_scores.append(f1)

        total += 1

    em_score = correct / total if total > 0 else 0
    avg_f1_score = np.mean(f1_scores) if f1_scores else 0
    return em_score, avg_f1_score

# Evaluate the teacher model and distilled model on FinQA
def evaluate_on_finqa():
    print("Evaluating on FinQA...")

    # Teacher model evaluation
    teacher_em, teacher_f1 = evaluate_qa_model(teacher_model, teacher_tokenizer, benchmark_finqa, task_config_finqa)
    print(f"Teacher Exact Match: {teacher_em:.4f}, Teacher F1: {teacher_f1:.4f}")

    # Distilled model evaluation
    distilled_em, distilled_f1 = evaluate_qa_model(distilled_model, distilled_tokenizer, benchmark_finqa, task_config_finqa)
    print(f"Distilled Exact Match: {distilled_em:.4f}, Distilled F1: {distilled_f1:.4f}")

    # Collect results
    results_data = [{
        "Workload": "FinQA",
        "Original Model Exact Match": teacher_em,
        "Distilled Model Exact Match": distilled_em,
        "Original Model F1": teacher_f1,
        "Distilled Model F1": distilled_f1,
        "Remarks": "QA task results"
    }]

    # Save results to CSV
    df = pd.DataFrame(results_data)
    df.to_csv("finqa_benchmark_results.csv", index=False)
    print("Evaluation completed. Results saved to 'finqa_benchmark_results.csv'")


if __name__ == "__main__":
    evaluate_on_finqa()