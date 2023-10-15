import torch
from transformers import LlamaTokenizerFast
from torch.utils.data import Dataset, DataLoader
import json
import time


def prepare_input(entry, has_labels=True):
    if has_labels:
        sent = entry['sent']
        tpls = []
        for tpl in entry['triples']:
            pred = tpl['pred']
            auxi = tpl['auxi']
            subj = tpl['subj']
            objs = tpl['objs']
            pred_str = ' ### '.join(auxi + [pred])

            obj_strs = []
            for obj in objs:
                curr_str = obj[1] if len(obj[0]) == 0 else f"{obj[0]} ### {obj[1]}"
                obj_strs.append(curr_str)
            tpl_str = ' ,, '.join([subj, pred_str] + obj_strs)
            tpls.append(tpl_str)
    else:
        sent = entry['s']
        ident = f"{entry['articleId']}_{entry['lineId']}"
        tpls = []
    
    instruction = f"You are an Open IE system to extract open relation triples. Format: \n"\
    "<subject> ,, <auxilliary ### predicate> ,, <prep1 ### object1> ,, <prep2 ### object2> ,, ... ."
    input_str = f"<s>[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\nExtract a list of open relation triples "\
f"from the following sentence: \n{sent} [/INST]\n1. "
    
    if has_labels:
        target_str = tpls[0]
        for i, tpl in enumerate(tpls):
            if i == 0:
                continue
            target_str += f"\n{i+1}. {tpl}"
        # print(f"input_str + target_str: {input_str + target_str}")
        
        return input_str, target_str
    else:
        return input_str, ident

class CaRBDataset(Dataset):
    def __init__(self, ifn, tokenizer, max_length=1024):
        self.entries = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(ifn, 'r', encoding='utf8') as ifp:
            for line in ifp:
                item = json.loads(line)
                # TODO: should we blend in CausalLM pretraining to mitigate forgetting?
                input_str, target_str = prepare_input(item, has_labels=True)
                self.entries.append((input_str, target_str))
    
    def __len__(self):
        return len(self.entries)
    
    # inspired by https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt#preparing-the-dataset
    def __getitem__(self, idx):
        prompt_str, target_str = self.entries[idx]
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        generation_ids = self.tokenizer.encode(target_str, add_special_tokens=False)

        full_input_ids = torch.tensor(prompt_ids+generation_ids)
        # print(full_input_ids)
        label_mask = torch.tensor([0]*len(prompt_ids)+[1]*len(generation_ids))
        block_mask = 1 - label_mask
        labels_input_ids = full_input_ids*label_mask - 100*block_mask

        # truncate to max length
        if full_input_ids.shape[0] > self.max_length-1:
            full_input_ids = full_input_ids[:self.max_length-1]
            labels_input_ids = labels_input_ids[:self.max_length-1]
        else:
            pass
        # Inspired by https://github.com/huggingface/transformers/issues/22794
        full_input_ids = torch.cat([full_input_ids, torch.tensor([self.tokenizer.eos_token_id])])
        labels_input_ids = torch.cat([labels_input_ids, torch.tensor([self.tokenizer.eos_token_id])])
        # print(f"full_input_ids.shape: {full_input_ids.shape}")
        return {
            'input_ids': full_input_ids,
            # 'attention_mask': torch.ones_like(full_input_ids), TODO: check
            'labels': labels_input_ids,
            'length': full_input_ids.shape[0]
        }


def collate_fn(batch, pad_method: str, tokenizer: LlamaTokenizerFast):
    # print(f"collate started")
    assert pad_method in ['bos', 'ign', 'pad']
    # pad input_ids and labels to the same lengths
    lengths = [len(item['input_ids']) for item in batch]
    assert all([len(item['labels']) == len(item['input_ids']) for item in batch])
    max_len = max(lengths)

    if pad_method == 'bos':
        pad_id = tokenizer.bos_token_id
        # pad left
        for item, curr_len in zip(batch, lengths):
            item['input_ids'] = torch.cat([torch.ones(max_len-curr_len, dtype=torch.long)*pad_id, item['input_ids']])
            item['labels'] = torch.cat([torch.ones(max_len-curr_len, dtype=torch.long)*-100, item['labels']])
    elif pad_method == 'ign':
        pad_id = -100
        # pad right
        for item, curr_len in zip(batch, lengths):
            item['input_ids'] = torch.cat([item['input_ids'], torch.ones(max_len-curr_len, dtype=torch.long)*pad_id])
            item['labels'] = torch.cat([item['labels'], torch.ones(max_len-curr_len, dtype=torch.long)*pad_id])
    elif pad_method == 'pad':
        pad = tokenizer.pad_token_id
        for item, curr_len in zip(batch, lengths):
            item['input_ids'] = torch.cat([item['input_ids'], torch.ones(max_len-curr_len, dtype=torch.long)*pad])
            item['labels'] = torch.cat([item['labels'], torch.ones(max_len-curr_len, dtype=torch.long)*-100])
    else:
        raise NotImplementedError

    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    # print(f"collate finished")
    return {
        'input_ids': input_ids,
        'labels': labels,
        # 'length': torch.tensor(lengths),
    }


class InferenceDataset(Dataset):
    def __init__(self, ifn, tokenizer, max_length=1024, has_labels=False):
        self.entries = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels
        with open(ifn, 'r', encoding='utf8') as ifp:
            for line in ifp:
                item = json.loads(line)
                # TODO: should we blend in CausalLM pretraining to mitigate forgetting?
                if has_labels:
                    input_str, target_str = prepare_input(item, has_labels=True)
                    self.entries.append((input_str, target_str))
                else:
                    input_str, ident = prepare_input(item, has_labels=False)
                    self.entries.append((input_str, ident))
    
    def __len__(self):
        return len(self.entries)
    
    # inspired by https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt#preparing-the-dataset
    def __getitem__(self, idx):
        if self.has_labels:
            prompt_str, target_str = self.entries[idx]
        else:
            prompt_str, ident = self.entries[idx]

        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False, return_tensors='pt')
        prompt_ids = prompt_ids['input_ids'].squeeze(0)
        if prompt_ids.shape[0] > self.max_length:
            prompt_ids = prompt_ids[:self.max_length]
        else:
            pass

        if self.has_labels:
            return {
                'input_ids': prompt_ids,
                'labels': target_str
            }
        else:
            return {
                'input_ids': prompt_ids,
                'ident': ident
            }



def inf_collate(batch, pad_method: str, tokenizer: LlamaTokenizerFast):
    assert pad_method in ['bos', 'ign', 'pad']
    # pad input_ids and labels to the same lengths
    lengths = [len(item['input_ids']) for item in batch]
    max_len = max(lengths)

    if pad_method == 'bos':
        pad_id = tokenizer.bos_token_id
        # pad left
        for item, curr_len in zip(batch, lengths):
            item['input_ids'] = torch.cat([torch.ones(max_len-curr_len, dtype=torch.long)*pad_id, item['input_ids']])
    elif pad_method == 'ign':
        pad_id = -100
        # pad right
        for item, curr_len in zip(batch, lengths):
            item['input_ids'] = torch.cat([item['input_ids'], torch.ones(max_len-curr_len, dtype=torch.long)*pad_id])
    elif pad_method == 'pad':
        pad = tokenizer.pad_token_id
        for item, curr_len in zip(batch, lengths):
            item['input_ids'] = torch.cat([item['input_ids'], torch.ones(max_len-curr_len, dtype=torch.long)*pad])
    else:
        raise NotImplementedError

    input_ids = torch.stack([item['input_ids'] for item in batch])
    if 'labels' in batch[0]:
        labels = [item['labels'] for item in batch]
        return {
            'input_ids': input_ids,
            'labels': labels,
            # 'length': torch.tensor(lengths),
        }
    elif 'ident' in batch[0]:
        idents = [item['ident'] for item in batch]
        return {
            'input_ids': input_ids,
            'idents': idents
        }
    else:
        raise ValueError


if __name__ == '__main__':
    tokenizer = LlamaTokenizerFast.from_pretrained('../lms/llama2-7b-chat-hf/')
    dataset = CaRBDataset('./CaRB/data/CaRBent_gold/dev_reanno.json', tokenizer, max_length=1024)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        print(batch['input_ids'].shape)
        print(batch['labels'].shape)
        print(batch['input_ids'][0])
        print(batch['labels'][0])
        break