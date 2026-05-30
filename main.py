import argparse
import sys

from BabyLM.tokenizer.bpe_tokenizer import BabyLMTokenizer
from BabyLM.tokenizer.visualize_trained_tokenizer import export_html_visualization
from BabyLM.data_handler.tokenize_corpus import tokenize_corpus, tokenize_from_raw
from BabyLM.train import add_pretrain_args, run_pretrain


def main():
    parser = argparse.ArgumentParser(description="BabyLM Training Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    p_tok = subparsers.add_parser("train-tokenizer", help="Train the tokenizer from scratch")
    p_tok.add_argument("--vocab-size", type=int, default=16384)
    p_tok.add_argument("--save-path", type=str, default="models/tokenizer.json")
    p_tok.add_argument("--visualize-tokenizer", action="store_true")

    p_corp = subparsers.add_parser("tokenize-corpus", help="Tokenize HF dataset to a flat uint16 bin")
    p_corp.add_argument("--tokenizer", type=str, default="models/gpt-bert-official.json")
    p_corp.add_argument("--output", type=str, default="data/train.bin")
    p_corp.add_argument("--dataset", type=str, default="BabyLM-community/BabyLM-2026-Strict-Small")
    p_corp.add_argument("--split", type=str, default="train")
    p_corp.add_argument("--text-column", type=str, default="text")
    p_corp.add_argument("--strip-speaker-tags", action="store_true",
                        help="remove CHILDES-style `*SPEAKER:\\t` prefix from each utterance")
    p_corp.add_argument("--insert-sep", action="store_true",
                        help="append [SEP] token between documents (utterance boundaries)")
    p_corp.add_argument("--source-mode", choices=["hf", "raw"], default="raw",
                        help="hf: load flattened HF dataset (one utterance per row); "
                             "raw: parse per-source txt files in --raw-dir, grouping utterances "
                             "into conversations/articles and inserting [SEP] only at those boundaries")
    p_corp.add_argument("--raw-dir", type=str, default="data/raw",
                        help="directory containing {childes,simple_wiki,gutenberg,open_subtitles,bnc_spoken,switchboard}.train.txt")
    p_corp.add_argument("--chunk-words", type=int, default=0,
                        help="raw mode: split delimiter-less sources (open_subtitles/bnc_spoken/"
                             "switchboard) into ~N-word pseudo-documents (<=0 = one doc per file, default)")

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
        if args.source_mode == "raw":
            tokenize_from_raw(
                tokenizer_path=args.tokenizer,
                output_path=args.output,
                raw_dir=args.raw_dir,
                strip_speaker_tags=args.strip_speaker_tags,
                chunk_words=args.chunk_words,
            )
        else:
            tokenize_corpus(
                tokenizer_path=args.tokenizer,
                output_path=args.output,
                dataset_name=args.dataset,
                split=args.split,
                text_column=args.text_column,
                strip_speaker_tags=args.strip_speaker_tags,
                insert_sep=args.insert_sep,
            )

    elif args.command == "pretrain":
        run_pretrain(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
