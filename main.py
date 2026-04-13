import argparse
import sys
from BabyLM.tokenizer.bpe_tokenizer import BabyLMTokenizer
from BabyLM.tokenizer.visualize_trained_tokenizer import export_html_visualization


def main():
    parser = argparse.ArgumentParser(description="BabyLM Training Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    parser_train_tok = subparsers.add_parser("train-tokenizer", help="Train the tokenizer from scratch")
    parser_train_tok.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size (default: 32000)")
    parser_train_tok.add_argument("--save-path", type=str, default="models/tokenizer.json", help="Path to save tokenizer (default: models/tokenizer.json)")
    parser_train_tok.add_argument("--visualize-tokenizer", action="store_true", help="Whether to run script that visualizes tokenizer model (default: False)")

    args = parser.parse_args()

    if args.command == "train-tokenizer":
        tokenizer_obj = BabyLMTokenizer(vocab_size=args.vocab_size, save_path=args.save_path)
        tokenizer_obj.train()
        if args.visualize_tokenizer:
            export_html_visualization("Well it's just that, you know, a pound, or a hundred pounds today, is not the same as a hundred pounds in a year's time, or two, two years' time.", tokenizer_obj.tokenizer)


    elif args.command == "train-model":
        print("Model training is not yet implemented.")
        # TODO: Implement model training routine

    elif args.command == "evaluate":
        print("Evaluation is not yet implemented.")
        # TODO: Implement evaluation routine
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
