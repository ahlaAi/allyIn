# Model Distillation and Fine-tuning

This project focuses on distilling large language models (LLMs) into smaller versions and fine-tuning them on domain-specific tasks. The distillation process uses a teacher model (`gpt-neo-1.3B`) to train a student model (`distilgpt2`). The model is then evaluated on several NLP benchmarks. Finally, it is finetuned on the Yahoo News Financial dataset and evaluated on FinQA. 

## Models:
- **Teacher Model**: `EleutherAI/gpt-neo-1.3B`
- **Student Model**: `distilgpt2`

## Frameworks:
- `transformers`
- `datasets`
- `torch`

## Results

Evaluation results will be saved in:
- `benchmark_results.csv` (Benchmark tasks)
- `finqa_benchmark_results.csv` (FinQA evaluation)