import torch
import argparse
from jsq.prune import joint_pq
from transformers import AutoTokenizer,AutoModelForCausalLM
from safetensors import torch as storch, safe_open
import json


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,  torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)
    with safe_open("/root/autodl-tmp/cache/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549/model-00001-of-00002.safetensors", framework="pt") as f:
        metadata1 = f.metadata()
        print("Metadata:", metadata1)
    with safe_open("/root/autodl-tmp/cache/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549/model-00002-of-00002.safetensors", framework="pt") as f:
        metadata2 = f.metadata()
        print("Metadata:", metadata2) 
    light_model = joint_pq(args, model, tokenizer)

    if args.save_model:
        with open('/root/autodl-tmp/results/model.safetensors.index.json', 'r') as f:
            index = json.load(f)
        state_dict = light_model.state_dict()
        json_dict =  index["weight_map"]
        part1 = {k: state_dict[k] for k in state_dict.keys() if json_dict[k] == "model-00001-of-00002.safetensors"}
        part2 = {k: state_dict[k] for k in state_dict.keys() if json_dict[k] == "model-00002-of-00002.safetensors"}
        storch.save_file(part1, args.save_model + "model-00001-of-00002.safetensors", metadata=metadata1)
        storch.save_file(part2, args.save_model + "model-00002-of-00002.safetensors", metadata=metadata2)
        print(f"模型已保存到 {args.save_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5, help='number of shots')
    parser.add_argument("--ngpu", "-g", type=int, default=8)
    parser.add_argument("--data_dir", "-d", type=str, default="data", required=True, help='dataset location')
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--path", type=str, required=False, help='model checkpoint location')

    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/cache", type=str)

    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--clip_l', type=float, default=0.0)
    parser.add_argument('--clip_h', type=float, default=0.01)
    parser.add_argument('--abs', action="store_false")
    parser.add_argument('--rho', type=float, default=2.1)
    parser.add_argument("--nbits", type=int, default=8)

    args = parser.parse_args()
    main(args)