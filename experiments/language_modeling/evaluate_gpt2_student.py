import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from utils.metrics import compute_perplexity


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 Student Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained student model")
    parser.add_argument("--task_name", type=str, default="wikitext", help="Evaluation task name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # 加载评估数据集
    if args.task_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    else:
        raise ValueError("Unsupported evaluation task")

    # 计算困惑度
    ppl = compute_perplexity(model, tokenizer, dataset, args.batch_size)
    print(f"Perplexity on {args.task_name} test set: {ppl}")


if __name__ == "__main__":
    main()
