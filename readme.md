This is the repository for the paper "Soft Prompt-tuning for Cyberbullying Detection".

**Firstly install OpenPrompt** https://github.com/thunlp/OpenPrompt

Then copy prompts/knowledgeable_verbalizer.py to Openprompt/openprompt/prompts/knowledgeable_verbalizer.py

example shell scripts:

**Perform a single training and testing session**
```shell
python fewshot.py --result_file ./result.txt --template_type soft --model_name_or_path google-bert/bert-base-uncased --dataset wiki --template_id 0 --seed 123 --batch_size 16 --shot 40 --learning_rate 2e-5 --max_epochs 10 --verbalizer kpt
```

**Automated training and testing**
```shell
python auto_run.py
```
Note that the file paths should be changed according to the running environment.
