import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from functools import partial
import time
import torch
from torch.utils.data import DataLoader
import json
import argparse
import sys
from optimum.gptq import GPTQQuantizer, load_quantized_model
import gc
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
                            GenerationConfig, EarlyStoppingCallback
from peft import PeftModel
from llamaOIE_utils import lora_wrap
# inf_collate is used in inference_llama2_peft (regular inference), inf_simple_collate is used in inference_llama_vllm (vllm inference, implemented w.r.t. llama3)
from llamaOIE_dataset import CaRBDataset, collate_fn, InferenceDataset, WikipediaDataset, inf_collate, inf_simple_collate
from llamaOIE_evalutils import compare_prediction_gold, parse_outstr_to_triples


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

        output_strs.append(ostr)

    del input_ids
    return output_strs


def train_llama_peft(args, model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)  #  , no_split_module_classes=['LlamaDecoderLayer']

    # tokenizer is used to pre-process the dataset, so is not directly involved in Trainer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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
        raise ValueError(f"Unknown peft type {args.peft_type}")

    train_fn = os.path.join(args.data_root, args.train_fn)
    dev_fn = os.path.join(args.data_root, args.dev_fn)
    test_fn = os.path.join(args.data_root, args.test_fn)
    
    dataset_train = CaRBDataset(train_fn, tokenizer, use_examples=args.use_examples, max_length=args.max_length)
    dataset_dev = CaRBDataset(dev_fn, tokenizer, use_examples=args.use_examples, max_length=args.max_length)
    dataset_test = CaRBDataset(test_fn, tokenizer, use_examples=args.use_examples, max_length=args.max_length)

    print(f"--> Training Set Length = {len(dataset_train)}")
    print(f"--> Dev Set Length = {len(dataset_dev)}")
    print(f"--> Test Set Length = {len(dataset_test)}")

    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.01)
    
    training_args = TrainingArguments(
        output_dir=args.ckpt_path,
        overwrite_output_dir=True,
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
        load_best_model_at_end=True,
        metric_for_best_model='loss',  # TODO: check
        greater_is_better=False,
        optim='adamw_torch_fused',
        group_by_length=True,  # TODO: check
        length_column_name='length',
        report_to=['tensorboard'],
        gradient_checkpointing=False,
        include_inputs_for_metrics=False,
    )

    collate_partial = partial(collate_fn, pad_method=args.pad_method, tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_partial,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        callbacks=[early_stop_callback],
    )
    torch._dynamo.config.verbose=True
    trainer.train()

    print(f"--> Finished training.")
    os.makedirs(os.path.join(args.ckpt_path, 'best_model'), exist_ok=True)
    model.save_pretrained(os.path.join(args.ckpt_path, 'best_model'))


def merge_trained_model(args, model_path, ckpt_path, merged_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)  #  , no_split_module_classes=['LlamaDecoderLayer']
    model = PeftModel.from_pretrained(model, ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    model = model.merge_and_unload()
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    print(f"Model merged and saved to {merged_path}.")

# @DeprecationWarning("This function is deprecated, use merge_trained_model instead.")
def merge_quantize_trained_model(args, model_path, ckpt_path, merged_path, merged_quantized_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)  #  , no_split_module_classes=['LlamaDecoderLayer']
    model = PeftModel.from_pretrained(model, ckpt_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.merge_and_unload()
    print(f"Model parameter names: {model.state_dict().keys()}")

    model.save_pretrained(merged_path)

    quantizer = GPTQQuantizer(bits=8, dataset="ptb", block_name_to_quantize = "model.layers", model_seqlen = 2048)  # LlamaDecoderLayer

    print(f"Beginning to Quantize model...")
    quantized_model = quantizer.quantize_model(model, tokenizer)
    print(f"Model Quantization completed.")

    quantizer.save(quantized_model, merged_quantized_path)


@torch.no_grad()
def inference_llama_regular(args, model_path, ckpt_path, merged_quantized_path, inference_fn, inference_id, evaluate: bool, write_to_file: bool,
                          use_quantized: bool, debug: bool, predict_wiki: bool, wiki_chunksize: int):
    if not evaluate and not write_to_file:
        print(f"Neither evaluation nor write_to_file is set, exiting...", file=sys.stderr)
        return
    print(f"Running inference for inference file: {inference_id}...")
    lemmatizer = WordNetLemmatizer()
    load_dtype = torch.bfloat16

    element_weights = {'subj': 1.0, 'pred': 1.0, 'auxi': 1.0, 'obj': 1.0}

    # tokenizer is used to pre-process the dataset, so is not directly involved in Trainer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if use_quantized:
        model = load_quantized_model(merged_quantized_path)
        model = model.to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=load_dtype, device_map="auto", max_memory=args.max_memory)  #  , no_split_module_classes=['LlamaDecoderLayer']
        print(f"Loading loRA checkpoint from {ckpt_path}...")
        model = PeftModel.from_pretrained(model, ckpt_path, torch_dtype=load_dtype)
    model.eval()
    model = torch.compile(model)

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
        max_new_tokens=args.max_new_tokens,
        # do_sample=False,  # TODO: check
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        bad_word_ids=[],
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=args.num_return_sequences,
    )

    if predict_wiki:
        eval_dataset = WikipediaDataset(int(inference_id), wiki_chunksize, tokenizer, use_examples=args.use_examples, max_length=args.max_length)
    else:
        eval_dataset = InferenceDataset(inference_fn, tokenizer, use_examples=args.use_examples, max_length=args.max_length,
                                        has_labels=evaluate)
    print(f"--> Inference Set Length = {len(eval_dataset)}")
    collate_partial = partial(inf_collate, pad_method=args.pad_method, tokenizer=tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_bsz, shuffle=False,
                                                    collate_fn=collate_partial)

    prediction_strs = []
    label_strs = []
    eval_dicts = []
    start_t = time.time()

    if write_to_file:
        out_fp = open(inference_fn + '_preds', 'w', encoding='utf8')
    else:
        out_fp = None

    for bidx, batch in enumerate(tqdm(eval_dataloader)):
        try:
            batch_output_strs = get_model_output(model, tokenizer, gen_configs, batch['input_ids'])
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"--> Batch {bidx} failed due to OOM, reducing batch size...", file=sys.stderr)

                # Reload model
                del model
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                print(torch.cuda.memory_summary())

                # tokenizer is used to pre-process the dataset, so is not directly involved in Trainer.
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if use_quantized:
                    model = load_quantized_model(merged_quantized_path)
                    model = model.to('cuda')
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=load_dtype, device_map="auto", max_memory=args.max_memory)  #  , no_split_module_classes=['LlamaDecoderLayer']
                    print(f"Loading loRA checkpoint from {ckpt_path}...")
                    model = PeftModel.from_pretrained(model, ckpt_path, torch_dtype=load_dtype)
                model.eval()
                model = torch.compile(model)

                input_ids_1 = batch['input_ids'][:len(batch['input_ids'])//2]
                input_ids_2 = batch['input_ids'][len(batch['input_ids'])//2:]
                batch_output_strs_1 = get_model_output(model, tokenizer, gen_configs, input_ids_1)
                batch_output_strs_2 = get_model_output(model, tokenizer, gen_configs, input_ids_2)
                batch_output_strs = batch_output_strs_1 + batch_output_strs_2

        if out_fp is not None:
            for inbatch_i, outstr in enumerate(batch_output_strs):
                cur_triples = parse_outstr_to_triples(outstr)
                if 'idents' in batch:
                    ident = batch['idents'][inbatch_i]
                    text = tokenizer.decode(batch['input_ids'][inbatch_i], skip_special_tokens=True)
                    out_line = json.dumps({'idents': ident, 'text': text, 'triples': cur_triples}, ensure_ascii=False)
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

            for inids, pred, label in zip(batch['input_ids'], batch_output_strs, batch['labels']):
                if debug:
                    print(f"Text: {tokenizer.decode(inids, skip_special_tokens=True)}")
                    print(f"Prediction: {pred}")
                    print(f"Label: {label}")
                    print('---')
                else:
                    pass
                entry_eval_dict = compare_prediction_gold(pred, label, lemmatizer=lemmatizer, element_weights=element_weights, 
                                                        tokenizer=tokenizer, f_score_beta=args.f_score_beta)
                eval_dicts.append(entry_eval_dict)
                print(entry_eval_dict)
    
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


@torch.no_grad()
def inference_llama_vllm(args, model_path, ckpt_path, inference_fn, inference_id, evaluate: bool, write_to_file: bool,
                          use_quantized: bool, debug: bool, predict_wiki: bool, wiki_chunksize: int):
    from vllm import LLM, SamplingParams
    
    if evaluate:
        raise NotImplementedError(f"vllm support not added for the CaRB dataset")
    elif not write_to_file:
        print(f"Neither evaluation nor write_to_file is set, exiting...", file=sys.stderr)
        return
    else:
        pass
    print(f"Running inference for inference file: {inference_id}...")

    if use_quantized:
        raise NotImplementedError(f"int8 quantized model not yet supported by vllm")
    else:
        model = LLM(model_path, dtype='bfloat16', enable_prefix_caching=True)
        tokenizer = model.get_tokenizer()

    assert args.pad_method == 'bos'
    assert args.num_beams == 1
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, top_p=args.top_p, n=args.num_return_sequences)

    if predict_wiki:
        eval_dataset = WikipediaDataset(int(inference_id), wiki_chunksize, tokenizer, use_examples=args.use_examples, max_length=args.max_length,
                                        return_tensor=False)
    else:
        eval_dataset = InferenceDataset(inference_fn, tokenizer, use_examples=args.use_examples, max_length=args.max_length,
                                        has_labels=evaluate, return_tensor=False)

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_bsz, shuffle=False, collate_fn=inf_simple_collate)
    
    print(f"--> Inference Set Length = {len(eval_dataset)}")
    
    _ = model.generate(prompt_token_ids=[eval_dataset[0]['input_ids']], sampling_params=sampling_params)
    start_t = time.time()

    out_fp = open(inference_fn + '_preds', 'w', encoding='utf8')

    for bidx, batch in enumerate(tqdm(eval_dataloader)):
        batch_outputs = model.generate(prompt_token_ids=batch['input_ids'], sampling_params=sampling_params)
        batch_outputs.sort(key=lambda x: int(x.request_id))
        batch_output_strs = [x.outputs[0].text for x in batch_outputs]

        for inbatch_i, outstr in enumerate(batch_output_strs):
            cur_triples = parse_outstr_to_triples(outstr)
            text = batch['sents'][inbatch_i]
            if 'idents' in batch:
                ident = batch['idents'][inbatch_i]
                out_line = json.dumps({'idents': ident, 'text': text, 'triples': cur_triples}, ensure_ascii=False)
            else:
                assert 'labels' in batch
                label = batch['labels'][inbatch_i]
                out_line = json.dumps({'labels': label, 'triples': cur_triples}, ensure_ascii=False)
            out_fp.write(out_line + '\n')
    
    durr = time.time() - start_t
    print(f"Inference GPU total run time: {durr//3600} hrs {(durr%3600)//60} m {(durr%3600)%60:.2f} s")
    out_fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str, default='')  # '/work/ec216/ec216/shared/lms'
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--model_midpath', type=str, default='')
    parser.add_argument('--task', type=str, default='peft')
    parser.add_argument('--machine', type=str, default=None, choices=['cirrus', 'gala1'])
    parser.add_argument('--use_examples', type=str, default=False, choices=['train', 'test'])

    parser.add_argument('--data_root', type=str, default='./CaRB/data/CaRBent_gold')
    parser.add_argument('--train_fn', type=str, default='dev_reanno.json')
    parser.add_argument('--dev_fn', type=str, default='test_1_reanno.json')
    parser.add_argument('--test_fn', type=str, default='test_2_reanno.json')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=256)

    parser.add_argument('--ckpt_path', type=str, default='./ckpts/%s_%s_%s_%e/')
    parser.add_argument('--train_bsz', type=int, default=2)
    parser.add_argument('--ga_steps', type=int, default=2)
    parser.add_argument('--eval_bsz', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pad_method', type=str, default='bos', choices=['bos', 'ign', 'pad'])

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

    parser.add_argument('--lora_r', type=int, default=16, help='rank in LoRA')
    parser.add_argument('--lora_alpha', type=int, default=32, help='alpha in LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='dropout in LoRA')
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--inference_nowrite', action='store_true')
    parser.add_argument('--inference_quantized', action='store_true')
    parser.add_argument('--inference_vllm', action='store_true')
    parser.add_argument('--predict_wiki', action='store_true')
    parser.add_argument('--wiki_chunksize', type=int, default=10000)

    args = parser.parse_args()
    assert args.task == 'predict' or not args.predict_wiki, "predict_wiki is only for task predict"

    args.ckpt_path = args.ckpt_path % (args.model_name, args.peft_type+f"{args.lora_r}", args.pad_method, args.lr)

    model_path = os.path.join(args.model_root, args.model_midpath, args.model_name)
    merged_path = os.path.join(args.ckpt_path, 'merged_model')
    merged_quantized_path = os.path.join(args.ckpt_path, 'merged_quantized_model')

    if args.machine == 'cirrus':
        args.max_memory = {0: "4GiB", 1: "14GiB", 2: "14GiB", 3: "14GiB"}
    elif args.machine == 'gala1':
        args.max_memory = {0: "20GiB", 1: "22GiB"}
    elif args.machine == None:
        args.max_memory = None
    else:
        raise ValueError(f"Unknown machine {args.machine}")

    if args.task == 'peft':
        train_llama_peft(args, model_path)
    elif args.task == 'evaluate':
        ckpt_path = os.path.join(args.ckpt_path, args.inf_ckpt_name)
        if args.eval_subset == 'dev':
            inf_fpath = os.path.join(args.data_root, args.dev_fn)
        elif args.eval_subset == 'test':
            inf_fpath = os.path.join(args.data_root, args.test_fn)
        else:
            raise ValueError(f"Unknown eval subset {args.eval_subset}")
        inference_llama_regular(args, model_path, ckpt_path=ckpt_path, merged_quantized_path=merged_path, inference_fn=inf_fpath, inference_id=args.eval_subset,
                              evaluate=True, write_to_file=not args.inference_nowrite, use_quantized=args.inference_quantized, debug=args.debug, 
                              predict_wiki=args.predict_wiki, wiki_chunksize=args.wiki_chunksize)
    elif args.task == 'merge':
        ckpt_path = os.path.join(args.ckpt_path, args.inf_ckpt_name)
        merge_trained_model(args, model_path, ckpt_path, merged_path)
    elif args.task == 'quantize':
        raise NotImplementedError(f"Quantization not yet supported.")
        # ckpt_path = os.path.join(args.ckpt_path, args.inf_ckpt_name)
        # merge_quantize_trained_model(args, model_path, ckpt_path, merged_path, merged_quantized_path)
    elif args.task == 'predict':
        args.inference_fn = args.inference_fn % args.inference_id
        ckpt_path = os.path.join(args.ckpt_path, args.inf_ckpt_name)
        inference_fpath = os.path.join(args.data_root, args.inference_fn)

        # When the inference dataset is wikipedia, the inference_fpath is only used for indicating the paths to store the predictions,
        # the actual input data are loaded from the huggingface cache directories.
        if args.inference_vllm:
            inference_llama_vllm(args, merged_path, ckpt_path, inference_fpath, inference_id=args.inference_id,
                                 evaluate=False, write_to_file=not args.inference_nowrite, use_quantized=args.inference_quantized, debug=args.debug,
                                 predict_wiki=args.predict_wiki, wiki_chunksize=args.wiki_chunksize)
        else:
            inference_llama_regular(args, model_path, ckpt_path=ckpt_path, merged_quantized_path=merged_quantized_path, inference_fn=inference_fpath, inference_id=args.inference_id,
                                evaluate=False, write_to_file=not args.inference_nowrite, use_quantized=args.inference_quantized, debug=args.debug, 
                                predict_wiki=args.predict_wiki, wiki_chunksize=args.wiki_chunksize)
    else:
        raise ValueError(f"Unknown task {args.task}")