from functools import partial
from typing import List

import fire
import torch.cuda
import transformers
import json
import argparse
import os
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from contextlib import nullcontext
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from peft import LoraModel, get_peft_model
from llama_oie_utils import lora_wrap, get_policies, fsdp_auto_wrap_policy, \
    clear_gpu_cache, setup_environ_flags, set_profiler
from config.fsdp import fsdp_config

def prepare_input(entry):
    sent = entry['sent']
    tpls = []
    for tpl in entry['triples']:
        pred = tpl['pred']
        auxi = tpl['auxi']
        subj = tpl['subj']
        objs = tpl['objs']
        pred_str = pred if len(auxi) == 0 else f"{auxi} ### {pred}"
        obj_strs = []
        for obj in objs:
            curr_str = obj[1] if len(obj[0]) == 0 else f"{obj[0]} ### {obj[1]}"
            obj_strs.append(curr_str)
        tpl_str = ' ,, '.join([subj, pred_str] + obj_strs)
        tpls.append(tpl_str)
    
    return sent, tpls


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



def train_llama2_peft(args, model_path):
    if args.enable_fsdp:
        dist.init_process_group("nccl")
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # model, tokenizer, gen_configs = get_model(model_path, args.max_new_tokens, args.machine, use_accelerate=not args.disable_accelerate,
                                                # peft_config=peft_config)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', no_split_module_classes=['LlamaDecoderLayer'])
    if args.enable_fsdp:  # TODO: check
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
    # print_model_size

    # tokenizer is used to pre-process the dataset, so is not directly involved in Trainer.
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # tokenizer.padding_side = 'left'
    # tokenizer.pad_token = tokenizer.bos_token
    # tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )
    model.resize_token_embeddings(model.config.vocab_size + 1)

    if args.peft_type == 'lora':
        model, peft_config = lora_wrap(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    else:
        pass

    if args.enable_fsdp:
        mixed_precision_policy, wrapping_policy = get_policies(args.mixed_precision, args.use_fp16, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy()
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy if args.peft_type is not None else wrapping_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gather=True,
            sync_module_states=False,  # True if args.low_cpu_fsdp else False,
            param_init_fn=None, # Also related to args.low_cpu_fsdp
        )
        policies.apply_fsdp_checkpointing(model)
    
    # TODO: load dataset
     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")
    
    if args.enable_profiler:
        profiler, profiler_cb = set_profiler()
    else:
        profiler = nullcontext()
    
    training_args = TrainingArguments(
        output_dir=args.ckpt_path,
        overwrite_output_dir=True,
        # bf16=True, 
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.eval_bsz,
        gradient_accumulation_steps=args.ga_steps,
        eval_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type='linear',  # as with default
        logging_dir='./logs',
        logging_strategy='steps',
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=3,
        dataloader_num_workers=4,
        disable_tqdm=False,
        # label_names=['labels'],
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        # fsdp=True,
        deepspeed='./config/deepspeed_config.json',
        optim='adamw_torch_fused',
        group_by_length=True,  # TODO: check
        length_column_name='len_input',
        report_to=['tensorboard'],
        # resume_from_checkpoint=None,
        gradient_checkpointing=False,
        include_inputs_for_metrics=False,
        auto_find_batch_size=True,
        torch_compile=True,
    )

    with profiler:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[profiler_cb] if args.enable_profiler else [],
        )

        trainer.train()

    model.save_pretrained(os.path.join(model_path, 'peft'))


def inference_llama2_peft(args, model_path):
    # peft_config = LoraConfig.from_pretrained(os.path.join(model_path, 'peft'))
    model, tokenizer, gen_configs = get_model(model_path, args.max_new_tokens, args.machine, use_accelerate=not args.disable_accelerate,
                                                peft_config=None)
    model = LoraModel.from_pretrained(model, os.path.join(model_path, 'peft'))
    model.eval()

    # TODO: do inference / evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str, default='/work/ec216/ec216/shared/lms')
    parser.add_argument("--model_name", type=str, default="llama2-7b-chat-hf")
    parser.add_argument('--task', type=str, default='interactive')
    parser.add_argument('--machine', type=str, default='cirrus', choices=['cirrus', 'gala1'])
    parser.add_argument('--max_new_tokens', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--disable_accelerate', action='store_true')

    parser.add_argument('--ckpt_path', type=str, default='./ckpts/%s_%s/')
    parser.add_argument('--enable_profiler', action='store_true')
    parser.add_argument('--train_bsz', type=int, default=8)
    parser.add_argument('--ga_steps', type=int, default=2)
    parser.add_argument('--eval_bsz', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)

    parser.add_argument('--peft_type', type=str, default=None, choices=['lora'])
    parser.add_argument('--enable_fsdp', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--use_fp16', action='store_true')

    parser.add_argument('--lora_r', type=int, default=8, help='rank in LoRA')
    parser.add_argument('--lora_alpha', type=int, default=32, help='alpha in LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='dropout in LoRA')



    args = parser.parse_args()
    args.ckpt_path = args.ckpt_path % (args.model_name, args.peft_type)

    model_path = os.path.join(args.model_root, args.model_name)
    print("Model loaded")

    if args.task == 'examples':
        model, tokenizer, gen_configs = get_model(model_path, args.max_new_tokens, args.machine, use_accelerate=not args.disable_accelerate,
                                                peft_config=None)
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
    elif args.task == 'peft':
        train_partial = partial(train_llama2_peft, args=args, model_path=model_path)
        fire.Fire(train_partial)
    else:
        raise ValueError(f"Unknown task {args.task}")


"""
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden 😱 What should I do? [/INST]
"""