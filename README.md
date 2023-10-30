# LLaMA2-based Open Information Extraction

## Overview

Our goal is to create a competitive Open IE model with the new generation of LLMs. Previous generations of Open IE models are low in coverage and/or purely extractive and thus cannot overcome even the simplest surface form variations.

We use the SOTA [LLaMA2](https://huggingface.co/docs/transformers/model_doc/llama2) model as our backbone, and employ LoRA-finetuning to teach the model to generate Open IE triples.

For training data we use the [CaRB dataset](https://github.com/dair-iitd/CaRB) as the basis, but convert the dataset into a format with richer information: 
`<subj> ,, (<auxi> ###) <predicate> ,, (<prep1> ###) <obj1>, (<prep2> ###) <obj2>, ...`

- `<subj>`: the subject of the triple
- `<auxi>`: the auxiliary of the triple (e.g. modal verbs, negations, temporal markers)
- `<predicate>`: the predicate of the triple
- objects:
    - `<prepX>`: the optional preposition corresponding to this object
    - `<objX>`: the object (this could include the direct objects, dative objects, as well as what was traditionally categorized as obliques)

The format conversion is done by querying [GPT-4](https://openai.com/research/gpt-4) on the CaRB dataset, with few-shot examples annotated by the author. The conversion process is described [here](./CaRB/data/data_conversion.md).

## Scripts

- ` srun -p gpu --gres gpu:2 --kill-on-bad-exit --qos gpu --pty bash -i `
- Train: ` CUDA_VISIBLE_DEVICES="6,7" nohup python -u llama2oie.py --model_root ../lms --model_name llama2-7b-chat-hf --task peft --machine gala1 --lr 5e-5 --pad_method bos --peft_type lora --lora_r 64 --num_epochs 12 > ./nhlogs/llama2oie_7bc_lora64_bos_5e-5_retry.log & ` 

- Evaluate: ` CUDA_VISIBLE_DEVICES="7" python -u llama2oie.py --machine gala1sgl --model_root ../lms --model_name llama2-7b-chat-hf --task evaluate --eval_subset test --peft_type lora --pad_method bos --lr 5e-5 --lora_r 64 --eval_bsz 8 --f_score_beta 0.5 --debug --use_16_full > ./nhlogs/llama2oie_7bc_lora64_bos_5e-5_eval_test_fp16.log & `
``
- Inference on News Corpora: ` CUDA_VISIBLE_DEVICES="6,7" nohup python -u llama2oie.py --model_root ../lms --model_name llama2-7b-chat-hf --task inference --data_root [YOUR_PATH_TO_NS/NC] --inference_fn [YOUR_NS/NC_FILENAME] --inference_id [NS/NC] --machine gala1 --peft_type lora --pad_method bos --lr 5e-5 --lora_r 64 --eval_bsz 4 > ./nhlogs/llama2oie_7bc_lora_[NS/NC].log & `
    - Example: ` CUDA_VISIBLE_DEVICES="7" nohup python -u llama2oie.py --machine gala1sgl --model_root ../lms --model_name llama2-7b-chat-hf --task predict --data_root ./newsspike/splits --inference_id 3 --peft_type lora --pad_method bos --lr 5e-5 --lora_r 64 --eval_bsz 8 --debug --use_16_full > ./inflogs/llama2oie_lora64_bos_5e-5_fp16_ns_3.log & `
    - bash: ` nohup bash run_nightly.sh 82 1 7 > ./inflogs/run_82.log & `

Hyper-parameters to tune:
- `--lr`: 1e-4 / 3e-5 / 1e-5
- `--lora_r`: 16 / 32 / **64**
- `--pad_method`: **bos** / pad (using an extra pad token causes fine-tuned model to output repetitive predictions)

Other options: 
- `--model_root`: path to the directory containing the cached model, should be left empty when caching to default directory with transformers
- `--model_name`: name of the cached model, default: `llama2-7b-chat-hf`
- `--task`: **peft** / evaluate / inference
- `--machine`: gala1 / cirrus (should do inference on cirrus with the GPU hours we got)
- `--data_root`: root directory to the training data / inference news data, depending on usage
- `--max_length`: max length of generation sequence, should not be too big to prevent process getting stuck at certain batch
- `--ckpt_path`: path to the LoRA checkpoint to be loaded.
- `--inf_ckpt_name`: name of the best checkpoint in the `ckpt_path` to be loaded for inference.
- `--inference_fn`: file-name of the file (in `data_root`) containing the news data to be inferred.
- `--inference_id`: Natural Language name (identifier) of the news data to be inferred.

If more than one node is required, exclusive mode --exclusive and --gres=gpu:4 options must be included in your submission script.

## Performance on Test2 set of CaRB (our test set)

| Model | Levenshtein Distance | Macro F-1 | Micro F-1 |
| --- | --- | --- | --- |
| LoRA LLaMA-7b | 5.85 | 50.21% | 44.58% |


Performance FP-16 (activations only):

average Levenshtein Distance: 5.856714178544636
------------------
Evaluation results for test:
Macro Precision: 0.5091859335622744
Macro Recall: 0.49692482990504494
Macro F-0.5: 0.5066855457273697
Macro F-1: 0.5029806709421882
------------------
Micro Precision: 0.45112961308637095
Micro Recall: 0.4409099173443718
Micro F-0.5: 0.4490479482274383
Micro F-1: 0.4459612239862576
------------------
Finished.

Performance Full FP-16 (model + activations):

average Levenshtein Distance: 5.860029940119761
------------------
Evaluation results for test:
Macro Precision: 0.5076758607844291
Macro Recall: 0.49679700222982387
Macro F-0.5: 0.5054621392471024
Macro F-1: 0.5021775202274847
------------------
Micro Precision: 0.4421115290553989
Micro Recall: 0.44048939463313813
Micro F-0.5: 0.4417861470937852
Micro F-1: 0.44129897118183364
------------------

## Link to LoRA-trained model

[Here](https://huggingface.co/Teddy487/LLaMA2-7b-for-OpenIE).

## Other requirements

Please use `python >= 3.7` to ensure correct behavior w.r.t item ordering.

## Bug log
- LLaMA2 generate not compatible with do_sample=True, causes error: `RuntimeError: probability tensor contains either inf, nan or element < 0`
- HuggingFace Trainer bug: when doing trainer.evaluate() with the test set, the trainer gets stuck after batch 23 / 53 (bsz 8), cpu usage stays at 100% with no progress. Problem does not exist when evaluating with the validation set, or doing evaluation manually without the trainer.