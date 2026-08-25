import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ckpt", type=str)
    parser.add_argument("output_ckpt", type=str)
    args = parser.parse_args()
    
    print(f"Loading {args.input_ckpt}...")
    ckpt = torch.load(args.input_ckpt, map_location="cpu")
    
    if "model_state_dict" in ckpt:
        print("Extracting model_state_dict...")
        new_ckpt = {"model_state_dict": ckpt["model_state_dict"]}
        torch.save(new_ckpt, args.output_ckpt)
        print(f"Saved inference-ready checkpoint to {args.output_ckpt}")
    else:
        print("No model_state_dict found in checkpoint.")

if __name__ == "__main__":
    main()
