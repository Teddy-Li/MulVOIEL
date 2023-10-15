import json
import random


random.seed(42)

entries = []
with open(f'./test_reanno.json', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        entries.append(item)

t1_size = len(entries) // 3 + 1
# t2_size = len(entries) - t1_size

random.shuffle(entries)
test_1_entries = entries[:t1_size]
test_2_entries = entries[t1_size:]

test_1_entries = sorted(test_1_entries, key=lambda x: x['sidx'])
test_2_entries = sorted(test_2_entries, key=lambda x: x['sidx'])

print(f"Test 1 size: {len(test_1_entries)}")
print(f"Test 2 size: {len(test_2_entries)}")

with open(f'./test_1_reanno.json', 'w', encoding='utf-8') as ofp:
    for item in test_1_entries:
        ofp.write(json.dumps(item, ensure_ascii=False) + '\n')

with open(f'./test_2_reanno.json', 'w', encoding='utf-8') as ofp:
    for item in test_2_entries:
        ofp.write(json.dumps(item, ensure_ascii=False) + '\n')

print("Done.")