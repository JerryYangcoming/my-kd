import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeBERTa Student Model for NLU Tasks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser.parse_args()


def main():
    args = parse_args()

    # Stage I: 训练过滤器
    subprocess.run(["bash", "scripts/run_stage1.sh", "--config", args.config])

    # Stage II: 训练学生模型
    subprocess.run(["bash", "scripts/run_stage2.sh", "--config", args.config])


if __name__ == "__main__":
    main()
