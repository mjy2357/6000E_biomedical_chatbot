import json
import os
records = []
input_direction = 'data/03.jsonl'
output_direction = 'data/03_med.json'
with open(input_direction, 'r') as file:
    for line in file:
        # 解析每一行的 JSON 数据
        data = json.loads(line)
        if data['meta']['pile_set_name'] in ["PubMed Abstracts", "PubMed Central"]:
            records.append(data['text'])
# for i in range(2):
#     print(records[i])
with open(output_direction, 'w', encoding='utf-8') as json_file:
    json.dump(records, json_file)
