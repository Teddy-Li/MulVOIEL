import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from transformers import LlamaTokenizerFast
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import json
import time
import spacy
from tqdm import tqdm


def prepare_input(entry, tokenizer, has_labels=True, use_examples=False):
    example_strs_train = [
        [
            "Earlier this year, President Bush made a final `` take - it - or - leave it '' offer on the minimum wage : an increase to $ 4.25 an hour over three years, and only if accompanied by a lower wage for the first six months of a job.",
            """1. President Bush ,,  made ,, a final `` take - it - or - leave it '' offer ,, on ### the minimum wage ,, earlier ### this year
2. Bush ,,  is ,, President
3. President Bush's final `` take - it - or - leave it '' offer on the minimum wage ,,  was an increase ,, to ### $ 4.25 an hour ,, over ### three years
4. President Bush's final `` take - it - or - leave it '' offer on the minimum wage ,,  was an increase ,, only if ### accompanied by a lower wage for the first six months of a job
5. a lower wage ,,  will be ,, for ### a job ,, in ### the first six months"""
        ],
        [
            "This finding indicated that organic compounds could carry current.",
            """1. This finding ,,  indicated ,, that ### organic compounds could carry current
2. organic compounds ,,  could carry ,, current"""
        ],
        [
            "However, Jesus is not accepted as the son by Muslims, who strictly maintain that he was a human being who was loved by God and exalted by God to ranks of the most righteous.",
            """1. Jesus ,,  not ### is accepted ,, by ### Muslims ,, as ### the son
2. Muslims ,,  strictly maintain that Jesus was ,, a human being
3. Muslims ,,  strictly maintain that Jesus was loved ,, by ### God
4. Muslims ,,  strictly maintain that Jesus was exalted ,, to ### ranks of the most righteous ,, by ### God
5. Jesus ,,  was ,, a human being
6. God ,,  loved ,, Jesus
7. God ,,  exalted ,, Jesus ,, to ### ranks of the most righteous"""
        ],
        [
            "After a short absence Keibler began a short feud with the evil Jillian Hall, which led to the two having a match on `` Velocity '', which Keibler lost.",
            """1. Keibler ,,  began ,, a short feud ,, with ### the evil Jillian Hall ,, after ### a short absence
2. a short feud with the evil Jillian Hall ,,  led to ,, the two having a match ,, on ### \"Velocity\"
3. Keibler ,,  lost ,, a match ,, on ### \"Velocity\""""
        ]
    ]

    # used exclusively for large-scale inference
    example_strs_test = [
        [
            "Burnham died of heart failure at the age of 86, on September 1, 1947 at his home in Santa, Barbara, California.",
            """1. Burnham ,,  died ,, of ### heart failure ,, at ### the age of 86 ,, on ### September 1 , 1947 ,, at ### his home in Santa Barbara , California
2. Burnham ,,  had ,, his home ,, in ### Santa Barbara , California
3. Santa Barbara ,,  is ,, in ### California"""
        ],
        [
            "Godzilla and Battra battled on the ocean floor, until they caused a rift to open between tectonic plates.",
            """1. Godzilla ,,  battled ,, Battra ,, on ### the ocean floor
2. Battra ,,  battled ,, Godzilla ,, on ### the ocean floor
3. Godzilla and Battra ,,  caused ,, a rift to open ,, between ### tectonic plates
4. Godzilla and Battra ,,  battled ,, on ### the ocean floor ,, until ### they caused a rift to open"""
        ],
        [
            "And he was in Ali's army in the Battle of Jamal and later it was Muhammad ibn Abu Bakr who escorted Aisha back to Madina.",
            """1. he ,,  was ,, in ### Ali 's army
2. he ,,  was ,, in ### the Battle of Jamal
3. Ali 's army ,,  was ,, in ### the Battle of Jamal
4. Muhammad ibn Abu Bakr ,,  escorted ,, Aisha ,, back ### to Madina ,, later
5. he ,,  was ,, Muhammad ibn Abu Bakr"""
        ],
        [
            "Byers states that global citizenship is a `` powerful term '' because `` people that invoke it do so to provoke and justify action, '' and encourages the attendees of his lecture to re-appropriate it in order for its meaning to have a positive purpose, based on idealistic values.",
            """1. Byers ,,  states ,, that ### global citizenship is a "powerful term" because "people that invoke it do so to provoke and justify action"
2. Byers ,,  encourages ,, the attendees of his lecture ,, to ### re-appropriate it in order for its meaning to have a positive purpose ,, based on ### idealistic values
3. global citizenship ,,  is ,, a "powerful term"
4. people ,,  invoke ,, it ,, to ### provoke and justify action
5. attendees of his lecture ,,  re-appropriate ,, it ,, in order for its meaning to have a positive purpose ,, based on ### idealistic values"""
        ]
    ]

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
    # Open relation tuples are predicate-argument structures from the original text.
    # There may be one, two or more relation tuples per sentence.
    # Extract all and only the relation tuples from the sentence.

    instruction = f"You are an Open IE system to extract open relation triples from text. Include diverse subjects, avoid repetition. Format: \n"\
    "<subject> ,, <auxilliary ### predicate> ,, <prep1 ### object1> ,, <prep2 ### object2> ,, ... ."

    if use_examples == 'train':
        example_strs = example_strs_train
    elif use_examples == 'test':
        example_strs = example_strs_test
    elif use_examples == False:
        example_strs = []
    else:
        raise ValueError(f"Invalid use_examples value: {use_examples}")

    example_strs.append([sent, None])

    example_strs = [[f"Extract a list of open relation triples from the following sentence: {s}", o] for s, o in example_strs]

    messages = [
        {'role': 'system',
         'content': instruction},
    ]
    for s, o in example_strs:
        messages.append({'role': 'user', 'content': s})
        if o is not None:
            messages.append({'role': 'assistant', 'content': o})

    input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_str += '1. '
    
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
    def __init__(self, ifn, tokenizer, use_examples, max_length=1024):
        self.entries = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(ifn, 'r', encoding='utf8') as ifp:
            for line in ifp:
                item = json.loads(line)
                # TODO: should we blend in CausalLM pretraining to mitigate forgetting?
                input_str, target_str = prepare_input(item, self.tokenizer, has_labels=True, use_examples=use_examples)
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
    def __init__(self, ifn, tokenizer, use_examples, max_length=1024, has_labels=False, return_tensor=True):
        self.entries = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels
        self.return_tensor = return_tensor
        with open(ifn, 'r', encoding='utf8') as ifp:
            for line in ifp:
                item = json.loads(line)
                # print(item)
                # TODO: should we blend in CausalLM pretraining to mitigate forgetting?
                if has_labels:
                    input_str, target_str = prepare_input(item, self.tokenizer, has_labels=True, use_examples=use_examples)
                    self.entries.append((input_str, target_str, item['s'] if 's' in item else item['sent']))
                else:
                    input_str, ident = prepare_input(item, self.tokenizer, has_labels=False, use_examples=use_examples)
                    self.entries.append((input_str, ident, item['s'] if 's' in item else item['sent']))
    
    def __len__(self):
        return len(self.entries)
    
    # inspired by https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt#preparing-the-dataset
    def __getitem__(self, idx):
        if self.has_labels:
            prompt_str, target_str, orig_sent = self.entries[idx]
        else:
            prompt_str, ident, orig_sent = self.entries[idx]

        if self.return_tensor:
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False, return_tensors='pt')
            prompt_ids = prompt_ids['input_ids'].squeeze(0)
            if prompt_ids.shape[0] > self.max_length:
                prompt_ids = prompt_ids[:self.max_length]
            else:
                pass
        else:
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)['input_ids']
            if len(prompt_ids) > self.max_length:
                prompt_ids = prompt_ids[:self.max_length]
            else:
                pass
        
        if self.has_labels:
            return {
                'input_ids': prompt_ids,
                'labels': target_str,
                'sent': orig_sent
            }
        else:
            return {
                'input_ids': prompt_ids,
                'ident': ident,
                'sent': orig_sent
            }



class WikipediaDataset(Dataset):
    def __init__(self, chunk_idx: int, chunk_size, tokenizer, use_examples, max_length=1024, return_tensor=True):
        self.entries = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensor = return_tensor

        start_i = chunk_idx*chunk_size
        end_i = (chunk_idx+1)*chunk_size

        # <id, url, title, text>
        self.raw_dataset = load_dataset("wikipedia", "20220301.en", split=f'train[{start_i}:{end_i}]')

        # print(f"Debugging!")
        # self.raw_dataset = self.raw_dataset.select(range(100))

        self.sentencizer = spacy.load('en_core_web_sm')
        self.sentencizer.add_pipe("sentencizer")
        self.sentencizer.select_pipes(enable=["sentencizer"])

        print(f"Sentencizing raw Wikipedia documents...")
        for raw_doc in tqdm(self.raw_dataset):
            doc = self.sentencizer(raw_doc['text'])
            for sidx, sent in enumerate(doc.sents):
                sent_dict = {
                    's': sent.text,
                    'articleId': raw_doc['id'],
                    'lineId': sidx
                }
                input_str, ident = prepare_input(sent_dict, self.tokenizer, has_labels=False, use_examples=use_examples)
                self.entries.append((input_str, ident, sent_dict['s']))
        print(f"Finished sentencizing raw Wikipedia documents. Total sentences: {len(self.entries)}")
    
    def __len__(self):
        return len(self.entries)
    
    # inspired by https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt#preparing-the-dataset
    def __getitem__(self, idx):
        prompt_str, ident, orig_sent = self.entries[idx]

        if self.return_tensor:
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False, return_tensors='pt')
            prompt_ids = prompt_ids['input_ids'].squeeze(0)
            if prompt_ids.shape[0] > self.max_length:
                prompt_ids = prompt_ids[:self.max_length]
            else:
                pass
        else:
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)['input_ids']
            if len(prompt_ids) > self.max_length:
                prompt_ids = prompt_ids[:self.max_length]
            else:
                pass
        
        return {
            'input_ids': prompt_ids,
            'ident': ident,
            'sent': orig_sent
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
    orig_sents = [item['sent'] for item in batch]
    if 'labels' in batch[0]:
        labels = [item['labels'] for item in batch]
        return {
            'input_ids': input_ids,
            'labels': labels,
            'sents': orig_sents
            # 'length': torch.tensor(lengths),
        }
    elif 'ident' in batch[0]:
        idents = [item['ident'] for item in batch]
        return {
            'input_ids': input_ids,
            'idents': idents,
            'sents': orig_sents
        }
    else:
        raise ValueError

def inf_simple_collate(batch):
    res = {
        'input_ids': [item['input_ids'] for item in batch],
        'sents': [item['sent'] for item in batch]
    }
    if 'labels' in batch[0]:
        res['labels'] = [item['labels'] for item in batch]
    elif 'ident' in batch[0]:
        res['idents'] = [item['ident'] for item in batch]
    else:
        raise ValueError
    return res


if __name__ == '__main__':
    raise NotImplementedError