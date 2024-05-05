import json
import random
input_direction = '/hpc2hdd/home/jmiao996/6000E_biomedical_chatbot/data/ai-medical-chatbot_train.json'
output_direction = 'data/pile_med_100000.json'
with open(input_direction, 'r') as file:
    data = json.load(file)
print(len(data))
# print(len(data[0]['conversations'][0]['value']))
# random_data = random.sample(data, 100000)
print(len(data))
# with open(output_direction, 'w', encoding='utf-8') as json_file:
#     json.dump(random_data, json_file)    
