import argparse
import sys

from BabyLM.tokenizer.bpe_tokenizer import BabyLMTokenizer
from BabyLM.tokenizer.visualize_trained_tokenizer import export_html_visualization
from BabyLM.data_handler.tokenize_corpus import tokenize_corpus
from BabyLM.train import add_pretrain_args, run_pretrain


def main():
    parser = argparse.ArgumentParser(description="BabyLM Training Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    p_tok = subparsers.add_parser("train-tokenizer", help="Train the tokenizer from scratch")
    p_tok.add_argument("--vocab-size", type=int, default=32000)
    p_tok.add_argument("--save-path", type=str, default="models/tokenizer.json")
    p_tok.add_argument("--visualize-tokenizer", action="store_true")

    p_corp = subparsers.add_parser("tokenize-corpus", help="Tokenize HF dataset to a flat uint16 bin")
    p_corp.add_argument("--tokenizer", type=str, default="models/tokenizer.json")
    p_corp.add_argument("--output", type=str, default="data/train.bin")
    p_corp.add_argument("--dataset", type=str, default="BabyLM-community/BabyLM-2026-Strict-Small")
    p_corp.add_argument("--split", type=str, default="train")
    p_corp.add_argument("--text-column", type=str, default="text")

    p_pre = subparsers.add_parser("pretrain", help="Pretrain GPT-BERT v0")
    add_pretrain_args(p_pre)

    args = parser.parse_args()

    if args.command == "train-tokenizer":
        tok = BabyLMTokenizer(vocab_size=args.vocab_size, save_path=args.save_path)
        tok.train()
        if args.visualize_tokenizer:
            export_html_visualization(
                "Well it's just that, you know, a pound, or a hundred pounds today, is not the same as a hundred pounds in a year's time, or two, two years' time.",
                tok.tokenizer,
            )

    elif args.command == "tokenize-corpus":
        tokenize_corpus(
            tokenizer_path=args.tokenizer,
            output_path=args.output,
            dataset_name=args.dataset,
            split=args.split,
            text_column=args.text_column,
        )

    elif args.command == "pretrain":
        run_pretrain(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
