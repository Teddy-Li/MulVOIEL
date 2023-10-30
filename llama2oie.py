from functools import partial
import time
import fire
import torch.cuda
from torch.utils.data import DataLoader
import torch
import json
import argparse
import os
import sys
import shutil
import glob
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from contextlib import nullcontext
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, \
                            GenerationConfig, EarlyStoppingCallback
from peft import LoraModel, get_peft_model, PeftModel
from llama_oie_utils import lora_wrap, set_profiler, compute_metrics
from config.fsdp import fsdp_config
from llama2oie_dataset import CaRBDataset, collate_fn, InferenceDataset, inf_collate
from llama2oie_evalutils import compare_prediction_gold, parse_outstr_to_triples
from torch.cuda.amp import autocast


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


def get_model_output(model, tokenizer, gen_configs, input_ids):
    torch.cuda.empty_cache()
    if isinstance(model, torch.nn.DataParallel):
        input_ids = input_ids.to('cuda')
    else:
        input_ids = input_ids.to('cuda:0')
    with torch.no_grad():
        output = model.generate(inputs=input_ids, generation_config=gen_configs)
    total_outlists = output.tolist()  # output.sequences.tolist()
    # net_scores = output.scores

    output_strs = []

    for inbatch_eidx in range(len(total_outlists)):
        this_net_outlist = []
        this_outlist = total_outlists[inbatch_eidx]
        for i in range(len(this_outlist)):
            if i < len(input_ids[inbatch_eidx]):
                assert this_outlist[i] == input_ids[inbatch_eidx][i].item()
            else:
                this_net_outlist.append(this_outlist[i])
        # assert len(this_net_outlist) <= len(net_scores), f"{len(this_net_outlist)} vs {len(net_scores)}"
        # this_net_scores = [net_scores[i][inbatch_eidx][this_net_outlist[i]].item() for i in
        #                    range(len(this_net_outlist))]
        # full_str = tokenizer.decode(this_outlist, skip_special_tokens=True)

        ostr = tokenizer.decode(this_net_outlist, skip_special_tokens=True)
        if len(this_net_outlist) > 510:
            print(f"Warning: output length {len(this_net_outlist)} exceeds 510!", file=sys.stderr)
            print(f"Output: {ostr}", file=sys.stderr)
        # print(f"Output: {ostr}")
        # print(f"Full Output: {full_str}")
        output_strs.append(ostr)

    del input_ids
    # print(f"Output: {output_str}")
    return output_strs


def train_llama2_peft(args, model_path):

    load_dtype = torch.float16 if args.use_16_full else torch.float32
    # model, tokenizer, gen_configs = get_model(model_path, args.max_new_tokens, args.machine, use_accelerate=not args.disable_accelerate,
                                                # peft_config=peft_config)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=load_dtype
                                             )  #  , no_split_module_classes=['LlamaDecoderLayer']

    # tokenizer is used to pre-process the dataset, so is not directly involved in Trainer.
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    if args.pad_method == 'bos':
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.bos_token_id
    elif args.pad_method == 'ign':
        tokenizer.pad_token_id = -100
    elif args.pad_method == 'pad':
        tokenizer.add_special_tokens(
            {

                "pad_token": "<PAD>",
            }
        )
        model.resize_token_embeddings(model.config.vocab_size + 1)
    else:
        raise ValueError(f"Unknown pad method {args.pad_method}")

    if args.peft_type == 'lora':
        model, peft_config = lora_wrap(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    else:
        pass

    train_fn = os.path.join(args.data_root, args.train_fn)
    dev_fn = os.path.join(args.data_root, args.dev_fn)
    test_fn = os.path.join(args.data_root, args.test_fn)
    
    # TODO: load dataset
     # Load and preprocess the dataset for training and validation
    dataset_train = CaRBDataset(train_fn, tokenizer, max_length=args.max_length)
    dataset_dev = CaRBDataset(dev_fn, tokenizer, max_length=args.max_length)
    dataset_test = CaRBDataset(test_fn, tokenizer, max_length=args.max_length)

    print(f"--> Training Set Length = {len(dataset_train)}")
    print(f"--> Dev Set Length = {len(dataset_dev)}")
    print(f"--> Test Set Length = {len(dataset_test)}")
    
    if args.enable_profiler:
        profiler, profiler_cb = set_profiler()
    else:
        profiler = nullcontext()

    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.01)
    
    bf16, fp16 = False, False
    if args.use_16:
        # TODO: unmask!
        # if torch.cuda.is_bf16_supported():
        #     bf16 = True
        # else:
        #     fp16 = True
        fp16 = True
    else:
        pass
    fp16_fulleval = fp16 and args.use_16_full
    bf16_fulleval = bf16 and args.use_16_full
    
    training_args = TrainingArguments(
        output_dir=args.ckpt_path,
        overwrite_output_dir=True,
        # bf16=bf16,
        # fp16=fp16,
        evaluation_strategy='steps',
        eval_steps=100,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.eval_bsz,
        gradient_accumulation_steps=args.ga_steps,
        eval_accumulation_steps=8,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type='linear',  # as with default
        logging_dir='./logs',
        logging_strategy='steps',
        logging_steps=30,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=2,
        dataloader_num_workers=0,  # TODO: check
        disable_tqdm=False,
        # label_names=['labels'],
        load_best_model_at_end=True,
        metric_for_best_model='loss',  # TODO: check
        greater_is_better=False,
        deepspeed=args.deepspeed,
        optim='adamw_torch_fused',
        group_by_length=True,  # TODO: check
        length_column_name='length',
        report_to=['tensorboard'],
        # resume_from_checkpoint=None,
        gradient_checkpointing=False,
        include_inputs_for_metrics=False,
        # auto_find_batch_size=True,
        # torch_compile=True,
    )

    collate_partial = partial(collate_fn, pad_method=args.pad_method, tokenizer=tokenizer)

    with profiler:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_partial,
            train_dataset=dataset_train,
            eval_dataset=dataset_dev,
            # compute_metrics=compute_metrics,
            callbacks=[profiler_cb] if args.enable_profiler else [] + [early_stop_callback],
        )
        torch._dynamo.config.verbose=True
        trainer.train()

        print(f"--> Finished training.")
        os.makedirs(os.path.join(args.ckpt_path, 'best_model'), exist_ok=True)
        model.save_pretrained(os.path.join(args.ckpt_path, 'best_model'))
        # test_predictions = trainer.predict(dataset_test)
        
    # model.save_pretrained(os.path.join(model_path, 'peft'))


def eval_llama2_peft(args, model_path, ckpt_path, eval_fn):
    training_args = TrainingArguments(
        output_dir=args.ckpt_path,
        # bf16=True, 
        per_device_eval_batch_size=args.eval_bsz,
        # eval_accumulation_steps=8,
        dataloader_num_workers=0,  # TODO: check
        disable_tqdm=False,
        # label_names=['labels'],
        metric_for_best_model='loss',  # TODO: check
        greater_is_better=False,
        deepspeed=args.deepspeed,
        optim='adamw_torch_fused',
        group_by_length=True,  # TODO: check
        length_column_name='length',
        report_to=['tensorboard'],
        # resume_from_checkpoint=None,
        include_inputs_for_metrics=False,
        fp16_full_eval=args.use_16_full,
        fp16=args.use_16,
        # auto_find_batch_size=True,
        # torch_compile=True,
    )

    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto',
                                             max_memory=args.max_memory)  #  , no_split_module_classes=['LlamaDecoderLayer']
    model = PeftModel.from_pretrained(model, ckpt_path)
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    collate_partial = partial(collate_fn, pad_method=args.pad_method, tokenizer=tokenizer)
    dataset_eval = CaRBDataset(eval_fn, tokenizer, max_length=args.max_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_partial,
        # compute_metrics=compute_metrics,
        # callbacks=[profiler_cb] if args.enable_profiler else [],
    )

    print(f"Evaluating on {eval_fn}...")

    trainer.evaluate(dataset_eval)

    pattern_to_delete = os.path.join(args.ckpt_path, 'checkpoint-*')
    for f in glob.glob(pattern_to_delete):
        shutil.rmtree(f)


def inference_llama2_peft(args, model_path, ckpt_path, inference_fn, inference_id, evaluate: bool, write_to_file: bool,
                          debug: bool):
    if not evaluate and not write_to_file:
        print(f"Neither evaluation nor write_to_file is set, exiting...", file=sys.stderr)
        return
    print(f"Running inference for inference file: {inference_id}...")
    lemmatizer = WordNetLemmatizer()
    # peft_config = LoraConfig.from_pretrained(os.path.join(model_path, 'peft'))
    load_dtype = torch.float16 if args.use_16_full else torch.float32

    if args.machine == 'gala1dp':
        # do data parallelism
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=load_dtype)
    else:
        model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto',
                                                max_memory=args.max_memory, torch_dtype=load_dtype)  #  , no_split_module_classes=['LlamaDecoderLayer']

    element_weights = {'subj': 1.0, 'pred': 1.0, 'auxi': 1.0, 'obj': 1.0}

    # tokenizer is used to pre-process the dataset, so is not directly involved in Trainer.
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    print(f"Loading loRA checkpoint from {ckpt_path}...")
    model = PeftModel.from_pretrained(model, ckpt_path, torch_dtype=load_dtype)
    model.eval()
    model = torch.compile(model)

    # if args.machine == 'gala1dp':
    #     model = torch.nn.DataParallel(model)
    # else:
    #     pass

    if args.pad_method == 'bos':
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.bos_token_id
    elif args.pad_method == 'ign':
        tokenizer.pad_token_id = -100
    elif args.pad_method == 'pad':
        tokenizer.add_special_tokens(
            {

                "pad_token": "<PAD>",
            }
        )
        model.resize_token_embeddings(model.config.vocab_size + 1)
    else:
        raise ValueError(f"Unknown pad method {args.pad_method}")

    gen_configs = GenerationConfig(
        max_new_tokens=args.max_length,
        # do_sample=False,  # TODO: check
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        bad_word_ids=[],
        num_return_sequences=args.num_return_sequences,
    )

    eval_dataset = InferenceDataset(inference_fn, tokenizer, max_length=args.max_length,
                                    has_labels=evaluate)
    print(f"--> Inference Set Length = {len(eval_dataset)}")
    collate_partial = partial(inf_collate, pad_method=args.pad_method, tokenizer=tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_bsz, shuffle=False,
                                                    collate_fn=collate_partial)
    
    print(f"FP-16 enabled flag: {args.use_16}")
    print(f"FP-16 full enabled flag: {args.use_16_full}")

    prediction_strs = []
    label_strs = []
    eval_dicts = []
    start_t = time.time()

    if write_to_file:
        out_fp = open(inference_fn + '_preds', 'w', encoding='utf8')
    else:
        out_fp = None

    for batch in tqdm(eval_dataloader):
        # if bidx % (1 if debug else 100) == 0:
        #     durr = time.time() - start_t
        #     print(f"--> Batch {bidx} / {len(eval_dataloader)} done; time elapsed = {durr // 3600} hrs {(durr % 3600) // 60} m {(durr % 3600) % 60:.2f} s")
        
        with autocast(enabled=args.use_16):
            batch_output_strs = get_model_output(model, tokenizer, gen_configs, batch['input_ids'])

        if out_fp is not None:
            for inbatch_i, outstr in enumerate(batch_output_strs):
                cur_triples = parse_outstr_to_triples(outstr)
                if 'idents' in batch:
                    ident = batch['idents'][inbatch_i]
                    out_line = json.dumps({'idents': ident, 'triples': cur_triples}, ensure_ascii=False)
                else:
                    assert 'labels' in batch
                    label = batch['labels'][inbatch_i]
                    out_line = json.dumps({'labels': label, 'triples': cur_triples}, ensure_ascii=False)
                out_fp.write(out_line + '\n')
        else:
            pass

        if evaluate:
            prediction_strs.extend(batch_output_strs)
            label_strs.extend(batch['labels'])

            for pred, label in zip(batch_output_strs, batch['labels']):
                if debug:
                    print(f"Prediction: {pred}")
                    print(f"Label: {label}")
                    print('---')
                else:
                    pass
                entry_eval_dict = compare_prediction_gold(pred, label, lemmatizer=lemmatizer, element_weights=element_weights, 
                                                        tokenizer=tokenizer, f_score_beta=args.f_score_beta)
                eval_dicts.append(entry_eval_dict)
                print(entry_eval_dict)
        # print(f"--> Batch {bidx} done")
    
    durr = time.time() - start_t
    print(f"Inference GPU total run time: {durr//3600} hrs {(durr%3600)//60} m {(durr%3600)%60:.2f} s")

    if evaluate:
        macro_prec = sum([x['prec'] for x in eval_dicts]) / len(eval_dicts)
        macro_rec = sum([x['rec'] for x in eval_dicts]) / len(eval_dicts)
        macro_f_score = (1 + args.f_score_beta ** 2) * macro_prec * macro_rec / (args.f_score_beta ** 2 * macro_prec + macro_rec)
        macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec)

        micro_prec = sum([x['prec_num'] for x in eval_dicts]) / sum([x['prec_den'] for x in eval_dicts])
        micro_rec = sum([x['rec_num'] for x in eval_dicts]) / sum([x['rec_den'] for x in eval_dicts])
        micro_f_score = (1 + args.f_score_beta ** 2) * micro_prec * micro_rec / (args.f_score_beta ** 2 * micro_prec + micro_rec)
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)

        levenshtein_distance_buckets = {}
        for x in eval_dicts:
            for d in x['levenshtein_dists']:
                if d not in levenshtein_distance_buckets:
                    levenshtein_distance_buckets[d] = 0
                levenshtein_distance_buckets[d] += 1
        match_sum = sum([k*v for k, v in levenshtein_distance_buckets.items()])
        match_cnt = sum([v for k, v in levenshtein_distance_buckets.items()])
        levenshtein_distance_buckets = {k: (v, f"{100*v/match_cnt:.2f}%") for k, v in sorted(levenshtein_distance_buckets.items(), key=lambda x: x[0])}
        print(f"Levenshtein Distance Buckets: {levenshtein_distance_buckets}")
        print(f"average Levenshtein Distance: {match_sum / match_cnt}")
        print(f"------------------")

        print(f"Evaluation results for {inference_id}:")
        print(f"Macro Precision: {macro_prec}")
        print(f"Macro Recall: {macro_rec}")
        print(f"Macro F-{args.f_score_beta}: {macro_f_score}")
        print(f"Macro F-1: {macro_f1}")
        print(f"------------------")
        print(f"Micro Precision: {micro_prec}")
        print(f"Micro Recall: {micro_rec}")
        print(f"Micro F-{args.f_score_beta}: {micro_f_score}")
        print(f"Micro F-1: {micro_f1}")
        print(f"------------------")
        print(f"Finished.")
    else:
        pass

    if out_fp is not None:
        out_fp.close()
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str, default='')  # '/work/ec216/ec216/shared/lms'
    parser.add_argument("--model_name", type=str, default="llama2-7b-chat-hf")
    parser.add_argument('--task', type=str, default='peft')
    parser.add_argument('--machine', type=str, default='cirrus', choices=['cirrus', 'gala1', 'cirrus_half', 'gala1dp', 'gala1sgl'])

    parser.add_argument('--data_root', type=str, default='./CaRB/data/CaRBent_gold')
    parser.add_argument('--train_fn', type=str, default='dev_reanno.json')
    parser.add_argument('--dev_fn', type=str, default='test_1_reanno.json')
    parser.add_argument('--test_fn', type=str, default='test_2_reanno.json')
    parser.add_argument('--max_length', type=int, default=256)

    parser.add_argument('--ckpt_path', type=str, default='./ckpts/%s_%s_%s_%e/')
    parser.add_argument('--enable_profiler', action='store_true')
    parser.add_argument('--train_bsz', type=int, default=2)
    parser.add_argument('--ga_steps', type=int, default=2)
    parser.add_argument('--eval_bsz', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pad_method', type=str, default='bos', choices=['bos', 'ign', 'pad'])
    parser.add_argument('--use_16', action='store_true')
    parser.add_argument('--use_16_full', action='store_true')

    parser.add_argument('--eval_subset', type=str, default='test', choices=['dev', 'test'])

    # the following block of arguments are for inference generation only.
    parser.add_argument('--inf_ckpt_name', type=str, default='best_model')
    parser.add_argument('--inference_fn', type=str, default='newsspike_%s.json', help='the inference file name, if set to None, will use the test set.')
    parser.add_argument('--inference_id', type=str, default='test')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--f_score_beta', type=float, default=1.0)

    parser.add_argument('--peft_type', type=str, default=None, choices=['lora'])
    parser.add_argument('--enable_fsdp', action='store_true')
    parser.add_argument('--deepspeed', action='store_true')

    parser.add_argument('--lora_r', type=int, default=16, help='rank in LoRA')
    parser.add_argument('--lora_alpha', type=int, default=32, help='alpha in LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='dropout in LoRA')
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--inference_nowrite', action='store_true')

    args = parser.parse_args()
    # + ('_half' if args.use_16_full else '')
    args.ckpt_path = args.ckpt_path % (args.model_name, args.peft_type+f"{args.lora_r}", args.pad_method, args.lr)
    if args.deepspeed:
        args.deepspeed = './config/deepspeed_config.json'
    else:
        args.deepspeed = None
    if args.use_16_full:
        args.use_16 = True
    model_path = os.path.join(args.model_root, args.model_name)

#     if args.task == 'examples':
#         model, tokenizer, gen_configs = get_model(model_path, args.max_new_tokens, args.machine, use_accelerate=not args.disable_accelerate,
#                                                 peft_config=None)
#         examples = [
#             """Extract relation triples from the following sentence:

# Given the discrepancy between sentence embedding and relation extraction, the original context is insufficient for demonstration retrieval.

# Answer:
# 1. 
# """,
# # 1. setence embedding - has discrepancy - relation extraction
# # 2.
#         ]
#         for exm in examples:
#             print(exm)
#             get_model_output(model, tokenizer, gen_configs, [exm])
    if args.task == 'peft':
        # train_partial = partial(train_llama2_peft, args=args, model_path=model_path)
        # fire.Fire(train_partial)
        train_llama2_peft(args, model_path)
    elif args.task == 'evaluate':
        if args.machine == 'cirrus':
            args.max_memory = {0: "4GiB", 1: "14GiB", 2: "14GiB", 3: "14GiB"}
        elif args.machine == 'cirrus_half':
            args.max_memory = {0: "13GiB", 1: "15GiB"}
        elif args.machine == 'gala1':
            args.max_memory = {0: "12GiB", 1: "22GiB"}  # , 2: "22GiB"}
            # args.max_memory = None
        elif args.machine == 'gala1dp':
            args.max_memory = None
        elif args.machine == 'gala1sgl':
            args.max_memory = {0: "20GiB"}
        else:
            raise ValueError(f"Unknown machine {args.machine}")
        
        ckpt_path = os.path.join(args.ckpt_path, args.inf_ckpt_name)

        if args.eval_subset == 'dev':
            inf_fpath = os.path.join(args.data_root, args.dev_fn)
        elif args.eval_subset == 'test':
            inf_fpath = os.path.join(args.data_root, args.test_fn)
        else:
            raise ValueError(f"Unknown eval subset {args.eval_subset}")
        inference_llama2_peft(args, model_path, ckpt_path=ckpt_path, inference_fn=inf_fpath, inference_id=args.eval_subset,
                              evaluate=True, write_to_file=not args.inference_nowrite, debug=args.debug)

    elif args.task == 'predict':
        if args.machine == 'cirrus':
            args.max_memory = {0: "4GiB", 1: "14GiB", 2: "14GiB", 3: "14GiB"}
        elif args.machine == 'cirrus_half':
            args.max_memory = {0: "13GiB", 1: "15GiB"}
        elif args.machine == 'gala1':
            args.max_memory = {0: "14GiB", 1: "15GiB"}  # , 2: "22GiB"}
            # args.max_memory = None
        elif args.machine == 'gala1dp':
            args.max_memory = None
        elif args.machine == 'gala1sgl':
            args.max_memory = {0: "20GiB"}
        else:
            raise ValueError(f"Unknown machine {args.machine}")

        # if args.inference_fn is None:
        #     args.inference_fn = args.test_fn
        args.inference_fn = args.inference_fn % args.inference_id
        ckpt_path = os.path.join(args.ckpt_path, args.inf_ckpt_name)
        inference_fpath = os.path.join(args.data_root, args.inference_fn)

        # eval_llama2_peft(args, model_path, ckpt_path, eval_fn=inference_fpath)

        inference_llama2_peft(args, model_path, ckpt_path=ckpt_path, inference_fn=inference_fpath, inference_id=args.inference_id,
                              evaluate=False, write_to_file=not args.inference_nowrite, debug=args.debug)
    else:
        raise ValueError(f"Unknown task {args.task}")