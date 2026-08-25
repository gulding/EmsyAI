import sys
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="EmsyAI CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    parser_train = subparsers.add_parser("train", help="Start training (V3 Titan by default)")
    
    # Chat command
    parser_chat = subparsers.add_parser("chat", help="Start the interactive chat REPL")
    parser_chat.add_argument("--base_checkpoint", type=str, default="checkpoints_v3/model_step_5000.pt")
    parser_chat.add_argument("--lora_checkpoint", type=str, default=None)
    parser_chat.add_argument("--tokenizer", type=str, default="dataset/tokenizer_v2.json")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from emsyai.training.train_v3 import main as train_main
        train_main()
    elif args.command == "chat":
        from emsyai.chat_instruct import chat_repl
        chat_repl(args.base_checkpoint, args.lora_checkpoint, args.tokenizer)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
