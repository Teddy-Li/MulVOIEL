from typing import List

import torch.cuda
import transformers
import json
import argparse
import os
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, GenerationConfig


def get_model(model_path, max_new_tokens, machine, use_accelerate=True):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    gen_configs = GenerationConfig(max_new_tokens=max_new_tokens, output_scores=True,
                                   return_dict_in_generate=True)
    configs = LlamaConfig.from_pretrained(model_path)
    if machine == 'cirrus':
        zero_mem = '7GIB'
        reg_mem = '14GIB'
    elif machine == 'gala1':
        zero_mem = '12GIB'
        reg_mem = '20GIB'
    else:
        raise ValueError(f"Unknown machine {machine}")

    if use_accelerate is True:
        accelerator = Accelerator()
        with init_empty_weights():
            model = LlamaForCausalLM._from_config(configs)
        num_gpus = torch.cuda.device_count()
        max_memory = {x: reg_mem for x in range(num_gpus)}
        max_memory[0] = zero_mem
        model = load_checkpoint_and_dispatch(model, model_path,
                                             device_map='auto',
                                             no_split_module_classes=['LlamaDecoderLayer'],
                                             max_memory=max_memory,
                                             offload_folder='/home/ec216/ec216/shared/lm_offload/')
        print(model.hf_device_map)
    else:
        model = LlamaForCausalLM.from_pretrained(model_path)
        model = model.to('cuda')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id

    return model, tokenizer, gen_configs


def get_model_output(model, tokenizer, gen_configs, in_strs: List[str]):
    input_toked = tokenizer(in_strs, return_tensors="pt")
    input_toked = input_toked.to('cuda:0')
    
    output = model.generate(input_toked.input_ids, generation_config=gen_configs)
    total_outlists = output.sequences.tolist()
    net_scores = output.scores
    for inbatch_eidx in range(len(total_outlists)):
        this_net_outlist = []
        this_outlist = total_outlists[inbatch_eidx]
        for i in range(len(this_outlist)):
            if i < len(input_toked['input_ids'][inbatch_eidx]):
                assert this_outlist[i] == input_toked['input_ids'][inbatch_eidx][i].item()
            else:
                this_net_outlist.append(this_outlist[i])
        assert len(this_net_outlist) <= len(net_scores), f"{len(this_net_outlist)} vs {len(net_scores)}"
        this_net_scores = [net_scores[i][inbatch_eidx][this_net_outlist[i]].item() for i in
                           range(len(this_net_outlist))]

    output_str = tokenizer.decode(output.sequences[0])

    print(f"Output: {output_str}")


def main(args):
    model_path = os.path.join(args.model_root, args.model_name)
    model, tokenizer, gen_configs = get_model(model_path, args.max_new_tokens, args.machine, use_accelerate=not args.disable_accelerate)
    print("Model loaded")

    if args.task == 'interactive':
        print(f"Type your prompt below. To exit, type 'exit'.")
        while True:
            prompt = input("Prompt: ")
            if prompt == 'exit':
                break
            get_model_output(model, tokenizer, gen_configs, [prompt])
    elif args.task == 'examples':
        examples = [
            """Extract relation triples from the following sentence:

Given the discrepancy between sentence embedding and relation extraction, the original context is insufficient for demonstration retrieval.

Answer:
1. 
""",
# 1. setence embedding - has discrepancy - relation extraction
# 2.
        ]
        for exm in examples:
            print(exm)
            get_model_output(model, tokenizer, gen_configs, [exm])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str, default='/work/ec216/ec216/shared/lms')
    parser.add_argument("--model_name", type=str, default="llama2-7b-chat-hf")
    parser.add_argument('--task', type=str, default='interactive')
    parser.add_argument('--machine', type=str, default='cirrus', choices=['cirrus', 'gala1'])
    parser.add_argument('--max_new_tokens', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--disable_accelerate', action='store_true')

    args = parser.parse_args()
    main(args)
