import transformers
import peft
import torch
from peft import LoraConfig, TaskType
from transformers import LlamaForCausalLM, LlamaTokenizerFast, TrainingArguments, Trainer


model_path = f'/home/tianyi/lms/llama2-7b-chat-hf'

tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)
prompt_toks = tokenizer.tokenize('hello world')
print(prompt_toks)
prompt_ids = tokenizer.encode('hello world', add_special_tokens=True)
generation_ids = tokenizer.encode('good morning', add_special_tokens=False)
full_input_ids = torch.tensor(tokenizer.build_inputs_with_special_tokens(prompt_ids, generation_ids))
label_mask = torch.tensor(tokenizer.create_token_type_ids_from_sequences(prompt_ids, generation_ids))
block_mask = 1 - label_mask
labels_input_ids = full_input_ids*label_mask - 100*block_mask

print(prompt_ids)
print(generation_ids)
print(full_input_ids)
print(label_mask)
print(block_mask)
print(labels_input_ids)