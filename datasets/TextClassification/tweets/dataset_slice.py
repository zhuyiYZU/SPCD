import csv
from sklearn.model_selection import train_test_split

# 读取数据
data = []
with open('output.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

    # 将数据分为特征和标签
X = [item[1] for item in data]  # 内容作为特征
y = [item[0] for item in data]  # 标签

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 将训练集保存到train.csv
with open('train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(zip(y_train, X_train))  # 写入数据行

# 将测试集保存到test.csv
with open('test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(zip(y_test, X_test))  # 写入数据行
