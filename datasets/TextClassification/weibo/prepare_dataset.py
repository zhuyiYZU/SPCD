import csv

# 将结果写入 CSV 文件
f = open('output-3.csv', 'a', newline='', encoding='utf-8')
writer = csv.writer(f)

p_n = 0
n_n = 0

# 打开 wb.txt 文件并读取内容
# weibo_media.txt, weibo_supplyment.txt
with open('dataset/BullyDataset/weibo_supplyment.txt', 'r', encoding='utf-8') as file:
    # 读取文件并分割每一行
    for line in file:
        # 使用制表符分割每一行
        parts = line.strip().split('\t')
        # 跳过表头行
        if parts[0] == 'time':
            continue

        # 初始化 annotators 的总和
        annotators = [0, 0, 0]  # 分别对应 annotator A, B, C

        # 获取每个annotators打分结果
        for i in range(2, 5):  # 跳过前三个字段：time, text, 和 annotator A
            annotators[i - 2] += int(parts[i])

        if sum(annotators) >= 3:
            result = 1
            p_n += 1
        else:
            result = 0
            n_n += 1

        writer.writerow([result, parts[1]])

f.close()

print(p_n, n_n)
