import csv

# 将结果写入 CSV 文件
f = open('output.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(f)

p_n = 0
n_n = 0

# 打开 wb.txt 文件并读取内容
# weibo_media.txt, weibo_supplyment.txt
with open('cyberbullying_tweets.csv', 'r', encoding='utf-8') as file:
    # 读取文件并分割每一行
    csv_reader = csv.reader(file, delimiter=',')
    for parts in csv_reader:
        # 跳过表头行
        if parts[0] == 'tweet_text':
            continue

        if parts[1] != "not_cyberbullying":
            result = 1
            p_n += 1
        else:
            result = 0
            n_n += 1

        writer.writerow([result, parts[0]])

f.close()

print(p_n, n_n)
