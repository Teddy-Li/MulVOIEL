# MulVOIEL: Multi-Valent Open IE with LLaMA

This repository presents a strong Open Information Extraction (Open IE) model, based on LLaMA LLMs and LoRA fine-tuning. The model is capable of extracting multi-valent relations (with >2 arguments).

## Get Started

### Using the Model

1. Install the relevant libraries (using the LLaMA3-8b-instruct model as an example):
    ```bash
    pip install transformers datasets peft torch
    ```
2. Load the model and perform inference (example):
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    from llamaOIE import parse_outstr_to_triples
    from llamaOIE_dataset import prepare_input

    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    peft_adapter_name = "Teddy487/LLaMA3-8b-for-OpenIE"

    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, peft_adapter_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    input_text = "Earlier this year , President Bush made a final `` take - it - or - leave it '' offer on the minimum wage"
    input_text, _ = prepare_input({'s': input_text}, tokenizer, has_labels=False)

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    outstr = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    triples = parse_outstr_to_triples(outstr)

    for tpl in triples:
        print(tpl)
    ```

### Training the Model

1. Clone the repository and install the requirements:
    ```bash
    git clone https://github.com/Teddy-Li/MulVOIEL
    cd MulVOIEL
    conda create -n MulVOIEL python=3.10
    pip install -r requirements.txt
    ```

2. Train the model (example):
    ```bash
    python -u llama2oie.py --model_root ../lms --model_name meta-llama/Meta-Llama-3-8B-Instruct --task peft --lr 5e-5 --pad_method bos --peft_type lora --lora_r 64 --num_epochs 12
    ```

    Please explore more options and hyper-parameters in the [script](llamaOIE.py), as well as the section below.

3. Evaluate the model (example):
    ```bash
    python -u llama2oie.py --model_root ../lms --model_name meta-llama/Meta-Llama-3-8B-Instruct --task evaluate --eval_subset test --peft_type lora --pad_method bos --lr 1e-4 --lora_r 64 --eval_bsz 8 --f_score_beta 0.5 --debug
    ```

## Overview

 Open IE is the task of extracting predicate-argument structures from text. While open IE has traditionally been formulated as a binary relation extraction task (outputting relation triples in the form of <span style="color:#2471A3">_subject_</span>, <span style="color:#A93226">_predicate_</span>, <span style="color:#138D75">_object_</span>), we extend this to multi-valent relation extraction, where the output can be a tuple of any length.
 In multi-valent relation tuples, <span style="color:#A93226">_predicates_</span> are essentially verbs and arguments are the <span style="color:#2471A3">_subjects_</span>, <span style="color:#138D75">_objects_</span>, and other <span style="color:#B7950B">_compl_</span><span style="color:#B9770E">_ements_</span>.

For example, given the following sentence:

 `Earlier this year , President Bush made a final `` take - it - or - leave it '' offer on the minimum wage`

 The following relation can be extracted:

 <<span style="color:#2471A3">President Bush</span>, <span style="color:#A93226">made</span>, <span style="color:#138D75">a final "take-it-or-leave-it" offer</span>, <span style="color:#B7950B ">on the minimum wage</span>, <span style="color:#B9770E">earlier this year</span>>


<!-- Previous generations of Open IE models are low in coverage and/or confined to the extraction of binary relations. Instead, our model is capable of extracting multi-valent relations, which are relations with more than one argument.  -->

We produce this multi-valent open IE system by fine-tuning SOTA LLMs with customized training data, using LoRA (low-rank adaptors) to train the model to generate Open IE tuples (__important!!__ these are not just triples, but can also be pairs or 4-tuples, etc.).

### Training Data

For training data, we use the [CaRB dataset](https://github.com/dair-iitd/CaRB) as the basis, but convert the dataset into a format with richer information: 
`<subj> ,, (<auxi> ###) <predicate> ,, (<prep1> ###) <obj1>, (<prep2> ###) <obj2>, ...`

- `<subj>`: the subject of the triple
- `<auxi>`: the auxiliary of the triple (e.g. modal verbs, negations, temporal markers)
- `<predicate>`: the predicate of the triple
- objects:
    - `<prepX>`: the optional preposition corresponding to this object
    - `<objX>`: the object (this could include the direct objects, dative objects, as well as what was traditionally categorized as obliques)

The format conversion is done by querying [GPT-4](https://openai.com/research/gpt-4) on the CaRB dataset, with few-shot examples annotated by the author. The conversion process is detailed [here](./CaRB/data/data_conversion.md).

The original CaRB dataset only contains dev and test sets. In order to train the model, we sub-split the original test set randomly into test1 and test2 sets, with a 1:2 size ratio. We use test1 as our dev set and test2 as our test set. The original dev set is used as our training set.

The statistics of the re-annotated dataset is as follows (consistent with the CaRB dataset):

| Subset | # of Sentences | # of Tuples |
| --- | --- | --- |
| dev (training set) | 641 | 2445 |
| test1 (dev set) | 212 | 809 |
| test2 (test set) | 422 | 1744 |

### Model

We use the SOTA [LLaMA2](https://huggingface.co/docs/transformers/model_doc/llama2) and [LLaMA3](https://huggingface.co/docs/transformers/main/en/model_doc/llama3) as the base models for our Open IE system. We fine-tune these models with LoRA (low-rank adaptors) to generate multi-valent Open IE tuples; training is conducted in bfloat16 precision. We list the prominent fine-tuning hyper-parameters in the below, and the full list of hyper-parameters can be found in the default values of the script.

- `lr`: 5e-5
- `lora_r`: 64
- `num_epochs`: 12

### Evaluation

Since the task is to generate multi-valent tuples instead of triples, we use the Levenshtein Distance as the primary evaluation metric. We also report the macro and micro F1 scores. The evaluation is done on the test2 set of the CaRB dataset. We report the performance of the models below.


| Model | Levenshtein Distance | Macro F-1 | Micro F-1 |
| --- | --- | --- | --- |
| [LoRA LLaMA2-7b](https://huggingface.co/Teddy487/LLaMA2-7b-for-OpenIE) | 5.85 | 50.21% | 44.58% |
| [LoRA LLaMA3-8b](https://huggingface.co/Teddy487/LLaMA3-8b-for-OpenIE) | 5.04 | 55.32% | 48.49% |


### Inference

Currently we support batched offline inference on the [NewsSpike dataset]() and the [Wikipedia corpus](https://huggingface.co/datasets/wikipedia). We also provide parsed data for the above corpora, processed with the [LoRA LLaMA3-8b](https://huggingface.co/Teddy487/LLaMA3-8b-for-OpenIE) model. We organise the inference results in the format of huggingface datasets, with each relation tuple per entry. The datasets include the following columns:

- articleId: the unique identifier of the article in the NewsSpike dataset or the document ID in the Wikipedia corpus
- lineId: the index of the sentence in the article or the document
- factId: the index of the fact in the sentence
- text: textualized fact (relation tuple)
- subj: the subject of the fact
- pred: the predicate of the fact
- auxi: the auxiliary of the fact
- prep1: the preposition of the first object
- obj1: the first object
- prep2: the preposition of the second object
- obj2: the second object
- prep3: the preposition of the third object
- obj3: the third object
- prep4: the preposition of the fourth object
- obj4: the fourth object
- prep5: the preposition of the fifth object
- obj5: the fifth object

Note that we allow up to 5 objects in the relation tuple. The datasets are stored in the following links: [NewsSpike dataset](), [Wikipedia corpus]().

In case you would like to perform inference on your own data, you can use the `predict` task in the script, with the `--data_root` and `--inference_fn` flags set to the path of your data and the filename of the data, respectively. You can pre-process your data to match the expected input format of `InferenceDataset` in the [script](llamaOIE_dataset.py).

## Scripts for Each Step


- Train Multi-Valent Open IE Models: 
    - LLaMA2 7B: ` python -u llama2oie.py --model_name meta-llama/Llama-2-7b-chat-hf --task peft --lr 5e-5 --pad_method bos --peft_type lora --lora_r 64 --num_epochs 12 `
    - LLaMA3 8B Instruct: ` python -u llama2oie.py --model_root ../lms --model_name meta-llama/Meta-Llama-3-8B-Instruct --task peft --lr 5e-5 --pad_method bos --peft_type lora --lora_r 64 --num_epochs 12 `

- Evaluate: 
    - LLaMA2 7B: ` python -u llama2oie.py --model_root ../lms --model_name meta-llama/Llama-2-7b-chat-hf --task evaluate --eval_subset test --peft_type lora --pad_method bos --lr 5e-5 --lora_r 64 --eval_bsz 8 --f_score_beta 0.5 --debug --use_16_full `
    - LLaMA3 8B Instruct: ` python -u llama2oie.py --model_root ../lms --model_name meta-llama/Meta-Llama-3-8B-Instruct --task evaluate --eval_subset test --peft_type lora --pad_method bos --lr 1e-4 --lora_r 64 --eval_bsz 8 --f_score_beta 0.5 --debug `


- Merge: ` python -u llama2oie.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --task merge --peft_type lora --pad_method bos --lr 5e-5 --lora_r 64 --debug `

<!-- - Merge + Quantize: ` CUDA_VISIBLE_DEVICES="4,5" nohup python -u llama2oie.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --task quantize --peft_type lora --pad_method bos --lr 1e-4 --lora_r 64 --debug > ./nhlogs/llama3-8b-instruct_lora64_1e-4_quantize.log & ` -->


- Batched Offline Inference: ` python -u llama2oie.py --model_name MODEL_NAME --task predict --data_root [YOUR_PATH_TO_NS] --inference_fn [YOUR_NS/NC_FILENAME] --inference_id InfID --peft_type lora --pad_method bos --lr 5e-5 --lora_r 64 --eval_bsz 10000 --inference_vllm `
    - Example with [NewsSpike dataset](): ` python -u llama2oie.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --task predict --data_root ./newsspike/splits --inference_id 0 --peft_type lora --pad_method bos --lr 5e-5 --lora_r 64 --eval_bsz 10000 --inference_vllm `
    <!-- - bash: ` nohup bash run_nightly.sh 236 3 1 > ./inflogs/run_236_238.log & ` -->
    - Example with [Wikipedia](https://huggingface.co/datasets/wikipedia): ` python -u llama2oie.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --task predict --data_root ./wikipedia_preds/splits --inference_fn wikipedia_en_20220301_%s.json --inference_id 0 --peft_type lora --pad_method bos --lr 5e-5 --lora_r 64 --predict_wiki --wiki_chunksize 10000 --eval_bsz 10000 --inference_vllm `
    <!-- - bash: ` nohup bash run_nightly_wiki.sh 641 10 7 > ./inflogs/runwiki_641_650.log & ` -->
    - Note: by default we use [VLLM](https://docs.vllm.ai/) for accelerated inference, but you can disable it by removing the `--inference_vllm` flag. With the `--inference_vllm` flag, the meaning of `--eval_bsz` changes to the number of entries to be processed in one chunk (VLLM will automatically decide the actual batch size, the default chunk size is 10000 to allow prediction results to be returned periodically).
    - Note: in order for VLLM backed inference to proceed smoothly, you need to first merge the LoRA weights provided above with the base model weights. This can be done by running the `merge` task above.

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


<!-- ## Performance on Test2 set of CaRB (our test set)

| Model | Levenshtein Distance | Macro F-1 | Micro F-1 |
| --- | --- | --- | --- |
| LoRA LLaMA-7b | 5.85 | 50.21% | 44.58% |
| LoRA LLaMA3-8b-instruct float32 trained bfloat16 eval 1e-4 | 5.1783 | 55.49% | 47.23% |
| LoRA LLaMA3-8b-instruct bfloat16 trained bfloat16 eval 1e-4 | 5.0351 | 54.55% | 46.68% |
| LoRA LLaMA3-8b-instruct bfloat16 trained bfloat16 eval 5e-5 | 5.0742 | 52.4% | 46.4% |
| LoRA LLaMA3-8b-instruct bfloat16 1e-4 TPLT train FewShot eval | 5.5191 | 46.09% | 39.37% |
| LoRA LLaMA3-8b-instruct bfloat16 1e-4 TPLT train ZeroShot eval | 4.9814 | 53.1% | 46.2% |
| LoRA LLaMA3-8b-instruct bfloat16 1e-4 TPLT train ZeroShot eval promptV3 | 5.4340 | 51.86% | 44.53% |
| LoRA LLaMA3-8b-instruct bfloat16 1e-4 TPLT train ZeroShot eval promptV4 | 5.3577 | 52.63% | 46.049% |
| LoRA LLaMA3-8b-instruct bfloat16 1e-4 TPLT train ZeroShot eval promptV1.5 | 4.8028 | 52.27% | 46.06% |
| LoRA LLaMA3-8b-instruct bfloat16 1e-4 TPLT train ZeroShot eval promptV5 | 5.1769 | 53.67% | 47.246% |
| LoRA LLaMA3-8b-instruct bfloat16 1e-4 TPLT train ZeroShot eval promptV5+\n | 5.1769 | 53.67% | 47.246% |
| LoRA LLaMA3-8b-instruct bfloat16 5e-5 TPLT train ZeroShot eval promptV5 | 5.0438 | 55.32% | 48.49% | -->
