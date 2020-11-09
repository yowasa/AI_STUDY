# Estimator

import tensorflow as tf

import pandas as pd

# 设置特征名称和标签名称
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# 加载数据集，并读取为Dataframe格式
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# header: 将header这一行指定为列名，并且从这一行开始记录数据，默认为header=0
# names:指定列名，如果文件中不包含header的行，应该显性表示header=None
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# 读取训练数据的前五行做观察
train.head()

# 将训练数据和测试数据中的标签去掉
train_y = train.pop('Species')
test_y = test.pop('Species')

# 标签列现已从数据中删除
train.head()


# 定义函数，将数据存储为dataset格式，可以节省内存，并且方便并行读取
# 此步是为了给搭建好的模型投喂格式正确的输入数据
def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # 将输入转换为数据集。
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # 如果在训练模式下混淆并重复数据。
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


# 特征列描述了如何使用输入。
# 指定模型应该如何解读特定特征的一种函数
my_feature_columns = []
for key in train.keys():
    print(key)
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# 构建一个拥有两个隐层，隐藏节点分别为 30 和 10 的深度神经网络
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 隐层所含结点数量分别为 30 和 10.
    hidden_units=[30, 10],
    # 模型必须从三个类别中做出选择。
    n_classes=3)


# 训练模型
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)


# 测试模型
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))