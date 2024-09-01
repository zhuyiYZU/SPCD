# -*- coding: utf-8 -*-
import logging
import subprocess
import time
from itertools import product
import csv

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # 配置日志记录器

    with open("result_table.csv", "w") as table:
        writer = csv.writer(table)
        writer.writerow(
            ['template', 'template_id', 'shot', 'max_epochs', 'lr', 'batch', 'acc', 'pre', 'recall', 'f1'])

    dataset = ['tweets']
    batch_sizes = {16}
    learning_rates = {'2e-5', '3e-5'}
    shots = {20, 30, 40}
    seeds = {123}
    template_id = {0, 1, 2, 3}
    max_epochs = {20, 10}
    verbalizer = {"kpt"}
    model_name_or_path = "google-bert/bert-base-uncased"
    for n, t, j, i, k, m, v, e in product(dataset, template_id, seeds, batch_sizes, learning_rates, shots, verbalizer, max_epochs):
        cmd = (
            f"python fewshot.py --result_file ./result.txt --template_type soft --model_name_or_path {model_name_or_path} "
            f"--dataset {n} --template_id {t} --seed {j} "
            f"--batch_size {i} --shot {m} --learning_rate {k} --max_epochs {e} --verbalizer {v}"
        )

        logging.info(f"Executing command: {cmd}")
        print(cmd)
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"Command executed successfully: {cmd}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")