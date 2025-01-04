import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset
from utils.metrics import compute_glue_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeBERTa Student Model on GLUE")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained student model")
    parser.add_argument("--task_name", type=str, required=True, help="GLUE task name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(
        'cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # 加载评估数据集
    dataset = load_dataset("glue", args.task_name, split="validation")

    # 计算评估指标
    results = compute_glue_metrics(model, tokenizer, dataset, args.task_name, args.batch_size)
    print(f"Evaluation results on {args.task_name}:")
    for metric, value in results.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
