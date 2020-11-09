import logging
import os
from collections import Counter

import numpy as np
import pandas as pd

# split data to 10 fold
fold_num = 10
data_file = '/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/train_set.csv'
f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')


# 读取num数量的数据 将其分为fold_num份 每一份等比拥有不同分类的数据
def all_data2fold(fold_num, num=10000):
    fold_data = []
    texts = f['text'].tolist()[:num]
    labels = f['label'].tolist()[:num]

    total = len(labels)

    index = list(range(total))
    np.random.shuffle(index)

    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        batch_size = int(len(data) / fold_num)
        other = len(data) - batch_size * fold_num
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)

        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)

    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))

    return fold_data


# 读取前1w条数据 将其分为十份
fold_data = all_data2fold(10, num=200000)

dir = './data/train_data'
for i in os.listdir(dir):
    file = os.path.join(dir, i)
    os.remove(file)
# 将数据储存至10个文件内 不同的文章用一个空行分开
for i in range(0, 10):
    data = fold_data[i]

    path = os.path.join(dir, "train_" + str(i))
    my_open = open(path, 'w')
    # 打开文件，采用写入模式
    # 若文件不存在,创建，若存在，清空并写入
    for text in data['text']:
        my_open.write(text)
        my_open.write('\n')  # 换行
        my_open.write('\n')  # 添加一个空行，作为文章之间的分隔符
    logging.info("complete train_" + str(i))
    my_open.close()

print('数据分组完成' + '*' * 80)

word_counter = Counter()
# 计算每个词出现的次数
data = f['text'].tolist()
for text in data:
    words = text.split()
    for word in words:
        word_counter[word] += 1

words = word_counter.keys()
path = os.path.join('./data', "vocab.txt")
my_open = open(path, 'w')

# 打开文件，采用写入模式
# 若文件不存在,创建，若存在，清空并写入
extra_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
my_open.writelines("\n".join(extra_tokens))
my_open.write("\n")
my_open.writelines("\n".join(words))
my_open.close()

print('字典创建完成' + '*' * 80)
