import tensorflow as tf
import numpy as np

print('产生一个大小为（5，2）的矩阵' + '*' * 100)
a = np.random.uniform(size=(5, 2))
print(a)

print('矩阵切分 将(5，2)切分为5个(2，)' + '*' * 100)
dataset = tf.data.Dataset.from_tensor_slices(a)
print(dataset)

for item in dataset:
    print(item)

# 对Dataset中的元素做变换
print('Dataset中的元素做map变换' + '*' * 100)
dataset_1 = dataset.map(lambda x: x + 1)

for item in dataset_1:
    print(item)

print('Dataset中的元素做batch变换 3个合成一个 不足3个则取剩下的合成一个' + '*' * 100)
dataset_2 = dataset_1.batch(3)

for item in dataset_2:
    print(item)

print('repeat操作 处理epoch' + '*' * 100)
for item in dataset_1.repeat(2):
    print(item)
# shuffle:打乱dataset中的元素，它有一个参数buffersize，表示打乱时使用的buffer的大小。
print('shuffle打乱操作' + '*' * 100)
for item in dataset_1.shuffle(buffer_size=3):
    print(item)

print('元组的切分' + '*' * 100)
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print(dataset3)

for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpy())

print('字典的切分' + '*' * 100)
dataset4 = tf.data.Dataset.from_tensor_slices({'feature': x,
                                               'label': y})
for item in dataset4:
    print(item['feature'].numpy(), item['label'].numpy())
