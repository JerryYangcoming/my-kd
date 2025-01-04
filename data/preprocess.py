import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess datasets for distillation")
    parser.add_argument("--task_name", type=str, required=True, help="GLUE task name or other")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--pad_to_max_length", action='store_true', help="Pad sequences to max length")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Directory to save processed data")
    return parser.parse_args()


def preprocess(args):
    if args.task_name in ['mnli', 'qqp', 'qnli', 'sst2', 'rte', 'cola', 'mrpc', 'stsb']:
        dataset = load_dataset("glue", args.task_name)
    elif args.task_name == 'squad':
        dataset = load_dataset("squad")
    else:
        raise ValueError("Unsupported task")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    def tokenize_function(examples):
        if args.task_name in ['mnli', 'qqp', 'qnli', 'sst2', 'rte', 'cola', 'mrpc', 'stsb']:
            if args.task_name == 'sst2':
                return tokenizer(examples['sentence'], padding='max_length' if args.pad_to_max_length else False,
                                 truncation=True, max_length=args.max_length)
            else:
                return tokenizer(examples['sentence1'], examples.get('sentence2', None),
                                 padding='max_length' if args.pad_to_max_length else False, truncation=True,
                                 max_length=args.max_length)
        elif args.task_name == 'squad':
            return tokenizer(examples['context'], examples['question'],
                             padding='max_length' if args.pad_to_max_length else False, truncation=True,
                             max_length=args.max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets.save_to_disk(os.path.join(args.output_dir, args.task_name))
    print(f"Preprocessed data saved to {os.path.join(args.output_dir, args.task_name)}")


if __name__ == "__main__":
    args = parse_args()
    preprocess(args)
