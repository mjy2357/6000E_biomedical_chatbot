import json
proxy = 'saves/LLaMA2-7B-Chat/lora'
input_dir_list = ['eval_on_medqa_test_aimediacalchatbotall','eval_on_medqa_test_pilemed100000','eval_on_pubmedqa_test_aimedicalchatbotall','eval_on_pubmedqa_test_pilemed100000']
input_file_name = 'generated_predictions.jsonl'
for input_dir in input_dir_list:
    path = proxy + '/' + input_dir + '/' + input_file_name
    with open(path, 'r') as file:
        count = 0
        correct = 0
        for line in file:
            count += 1
            sample = json.loads(line)
            label = sample['label']
            predict = sample['predict']
            if label[:2] in predict:
                correct += 1
    acc = correct / count if count != 0 else 0
    print(f'The accuracy of {input_dir} is {acc:.2%}')