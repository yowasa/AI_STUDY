import logging
import random

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
# set seed
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# 设置运行环境 使用cpu还是gpu运算
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

# split data to 10 fold
fold_num = 10
data_file = '/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/train_set.csv'
import pandas as pd


# 读取num数量的数据 将其分为fold_num份 每一份等比拥有不同分类的数据
# 最终结构为
def all_data2fold(fold_num, num=10000):
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
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


# 读取前1w条数据 将其分为十分
fold_data = all_data2fold(10)

# 开始构建训练测试集数据
fold_id = 9

# 最后一份数据为验证集数据
dev_data = fold_data[fold_id]

# 前面的9份数据为训练数据 将其整合在一起
train_texts = []
train_labels = []
for i in range(0, fold_id):
    data = fold_data[i]
    train_texts.extend(data['text'])
    train_labels.extend(data['label'])
# 最终的训练数据集
train_data = {'label': train_labels, 'text': train_texts}

# 测试数据集（需要提交的数据集）
test_data_file = '/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/test_a.csv'
f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
texts = f['text'].tolist()
# 将其label设为0 转化为注重的测试数据集
test_data = {'label': [0] * len(texts), 'text': texts}

# 开始构建词典
from collections import Counter


# 词典类
class Vocab():
    # 创建词典对象
    def __init__(self, train_data):
        # 最低词频 低于词词频不计数
        self.min_count = 5
        # pad标识占用的id
        self.pad = 0
        # unk标识占用的id
        self.unk = 1
        # 统计到的词集合 index即为编码的id 词频大于min_count的词的集合 [PAD]记为填充 在将句子补齐成同一维度时使用 [UNK]记为未识别 在词典中找不到的词就用此词
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']

        self._id2label = []
        self.target_names = []
        # 构建词典
        self.build_vocab(train_data)

        reverse = lambda x: dict(zip(x, range(len(x))))
        # 构建词与id的反向映射
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)

        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    # 使用训练数据集构建词典
    def build_vocab(self, data):
        # 构建词典时使用的Counter对象
        self.word_counter = Counter()
        # 拿到其中的text数据 进行词频统计 -为什么不直接套Counter?
        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1

        # 统计结束后将词频大于min_count的词放入_id2word进行编码
        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        # 标签的含义
        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        # 标签的Counter对象
        self.label_counter = Counter(data['label'])

        # 构建id的编码 -感觉无意义 且代码写的 emmm 很有问题 count这句是不需要的 甚至counter本身也是不需要的
        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])

    # 加载已经训练好emb向量
    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1

        embeddings[self.unk] = embeddings[self.unk] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        assert len(set(self._id2extword)) == len(self._id2extword)

        return embeddings

    # 词转id
    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    # 词向量转id
    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)

    # 标签转id
    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    # 词数量
    @property
    def word_size(self):
        return len(self._id2word)

    # 词向量数量
    @property
    def extword_size(self):
        return len(self._id2extword)

    # 标签大小
    @property
    def label_size(self):
        return len(self._id2label)


# 构建字典
vocab = Vocab(train_data)

# 构建模型
import torch.nn as nn
import torch.nn.functional as F


# 注意力模型
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 生成注意力权重 隐藏节点*隐藏节点的矩阵 代表每个句向量与其他句向量的关系
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # 均值为0 标准差为0.05
        self.weight.data.normal_(mean=0.0, std=0.05)
        # 将tensor转化为可训练的参数
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))
        #
        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # 一个线性模型 matmul为张量乘法 隐藏节点*attention权重矩阵+偏置 得到背景变量（由于并不是decoder相乘故为自注意力机制）
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # b x len x hidden

        # 完成注意力机制加权 转换后的张量与query相乘（线性变换）
        outputs = torch.matmul(key, self.query)  # b x len
        # 进行mask填充 0不变 1变为fill第二个value值 由于我们定义的mask为1是不变 故进行1-转换 替换值为一个大的负值以免影响softmax
        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))
        # 进行softmax 获得注意力得分
        attn_scores = F.softmax(masked_outputs, dim=1)  # b x len

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # sum weighted sources bmm为矩阵乘法 注意力得分与计算得到的key相乘 降维 得到最终结果
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden

        return batch_outputs, attn_scores


# build word encoder
word2vec_path = '../emb/word2vec.txt'
dropout = 0.15


# encoder编码类
class WordCNNEncoder(nn.Module):
    def __init__(self, vocab):
        super(WordCNNEncoder, self).__init__()
        # Dropout模型
        self.dropout = nn.Dropout(dropout)
        # 句向量维度
        self.word_dims = 100
        # 词id的Embedding模型
        self.word_embed = nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)
        # 词向量加载预先训练好的词向量
        extword_embed = vocab.load_pretrained_embs(word2vec_path)
        # 词向量参数 extword_size为词向量数量 word_dims为词向量维度
        extword_size, word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." % (extword_size, word_dims))

        # 词向量的Embedding模型
        self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

        input_size = self.word_dims
        # 卷积层
        self.filter_sizes = [2, 3, 4]  # n-gram window
        self.out_channel = 100
        # 卷积模型列表
        self.convs = nn.ModuleList([nn.Conv2d(1, self.out_channel, (filter_size, input_size), bias=True)
                                    for filter_size in self.filter_sizes])

    def forward(self, word_ids, extword_ids):
        # word_ids: sen_num x sent_len
        # extword_ids: sen_num x sent_len
        # batch_masks: sen_num x sent_len
        sen_num, sent_len = word_ids.shape
        # 对词和词向量进行embeding操作
        word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
        extword_embed = self.extword_embed(extword_ids)
        # 将embeding结果相加
        batch_embed = word_embed + extword_embed

        # 丢弃15%
        if self.training:
            batch_embed = self.dropout(batch_embed)

        # 升维操作 用来卷积
        batch_embed.unsqueeze_(1)  # sen_num x 1 x sent_len x 100
        # (2*100)*3
        pooled_outputs = []
        for i in range(len(self.filter_sizes)):
            # 长度-filtersize+1
            filter_height = sent_len - self.filter_sizes[i] + 1
            # 执行卷积模型
            conv = self.convs[i](batch_embed)
            # 进行非线性激活 得到隐藏层节点
            hidden = F.relu(conv)  # sen_num x out_channel x filter_height x 1
            # 池化层 最终池化为两个数据(1,2,100)
            mp = nn.MaxPool2d((filter_height, 1))  # (filter_height, filter_width)
            pooled = mp(hidden).reshape(sen_num,
                                        self.out_channel)  # sen_num x out_channel x 1 x 1 -> sen_num x out_channel
            # 池化输出结果加入集合
            pooled_outputs.append(pooled)
        # 将池化输出结果拼接在一起
        reps = torch.cat(pooled_outputs, dim=1)  # sen_num x total_out_channel
        # 经过过dropout
        if self.training:
            reps = self.dropout(reps)

        return reps


# build sent encoder
sent_hidden_size = 256
sent_num_layers = 2


class SentEncoder(nn.Module):
    def __init__(self, sent_rep_size):
        super(SentEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # LSTM模型
        self.sent_lstm = nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len
        # 进入LSTM模型
        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        # 进行mask mask升维 全部为1所以无改变
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)
        # dropout
        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens


# 最终模型类
class Model(nn.Module):
    # 用字典初始化
    def __init__(self, vocab):
        super(Model, self).__init__()
        self.sent_rep_size = 300
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}
        parameters = []
        # 编码模型 embeding
        self.word_encoder = WordCNNEncoder(vocab)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))
        # 发射模型 lstm
        self.sent_encoder = SentEncoder(self.sent_rep_size)
        # 注意力模型
        self.sent_attention = Attention(self.doc_rep_size)
        # 模型参数增加 发射模型 和注意力模型的参数
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))
        # 输出模型 一个线性模型 第一层是发射模型中lstm模型节点的两倍(接收发射模型的参数) 第二个为词典标签大小 有偏置
        self.out = nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
        # 将输出模型的参数增加到模型参数中
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters

        logging.info('Build model with cnn word encoder, lstm sent encoder.')

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    # 开始执行模型训练 batch_inputs为训练数据
    def forward(self, batch_inputs):
        # 提取需要的数据
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        # 先进行词编码 input1:词编码矩阵 input2:词向量矩阵 得到了句向量的特征
        sent_reps = self.word_encoder(batch_inputs1, batch_inputs2)  # sen_num x sent_rep_size
        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)  # b x doc_len x sent_rep_size
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)  # b x doc_len x max_sent_len
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        # 再进入发射模型 sent_reps为编码后的encoder矩阵 sent_masks为值为1的矩阵
        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)  # b x doc_len x doc_rep_size

        # 再进入注意力模型 sent_hiddens为发射模型的隐状态 sent_masks为值为1的矩阵
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)  # b x doc_rep_size

        # 最后进入输出矩阵
        batch_outputs = self.out(doc_reps)  # b x num_labels

        # 返回输出值
        return batch_outputs


model = Model(vocab)

# build optimizer
learning_rate = 2e-4
decay = .75
decay_step = 1000


# 优化器函数
class Optimizer:
    def __init__(self, model_parameters):
        self.all_params = []
        self.optims = []
        self.schedulers = []

        for name, parameters in model_parameters.items():
            if name.startswith("basic"):
                # 使用Adam进行优化
                optim = torch.optim.Adam(parameters, lr=learning_rate)
                self.optims.append(optim)

                l = lambda step: decay ** (step // decay_step)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=l)
                self.schedulers.append(scheduler)
                self.all_params.extend(parameters)

            else:
                Exception("no nameed parameters.")

        self.num = len(self.optims)

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = ' %.5f' * self.num
        res = lr % lrs
        return res


# 将文档text 依据词典vocab 转化为每段大小为max_sent_len的最多max_segment段
def sentence_split(text, vocab, max_sent_len=256, max_segment=16):
    words = text.strip().split()
    document_len = len(words)
    # 从0到文档长度 步长为max_sent_len的列表
    index = list(range(0, document_len, max_sent_len))
    # 追加文档最大长度
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        # [[长度,内容],[长度,内容]]
        segments.append([len(segment), segment])

    assert len(segments) > 0
    # 超出的部分截断 （逻辑为保留开头和结尾)通常认为这种做法会比较能体现文章主题
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        return segments


# 获取模型使用的数据 data为原数据集 vocab为字典
def get_examples(data, vocab, max_sent_len=256, max_segment=8):
    label2id = vocab.label2id
    examples = []

    for text, label in zip(data['text'], data['label']):
        # 通过lable获取到lable的id(对于现有的数据形式着实没必要)
        id = label2id(label)

        # 将文章转化成数据集 即每篇文章分段 每段最长长度 返回的是[[长度,内容]]
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            # 内容依据词典转化为id
            word_ids = vocab.word2id(sent_words)
            # 内容依据词典转化为词向量
            extword_ids = vocab.extword2id(sent_words)
            # doc结构为[段落长度,词id,词向量]
            doc.append([sent_len, word_ids, extword_ids])
        # example为 [lable_id,doc长度（多少段）,doc][段落长度,词id,词向量]
        examples.append([id, len(doc), doc])

    logging.info('Total %d docs.' % len(examples))
    return examples


# 数据切片 每调用一次返回一个切片的数据
def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield docs


# 数据迭代 每调用一次返回一个迭代批次的数据
def data_iter(data, batch_size, shuffle=True, noise=1.0):
    batched_data = []
    if shuffle:
        # 先打乱
        np.random.shuffle(data)
        # 获得所有文本长度信息
        lengths = [example[1] for example in data]
        # 增加+-1之间的偏差
        noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
        # 下标排序
        sorted_indices = np.argsort(noisy_lengths).tolist()
        # 长度从小到大输出 让同样长度的文本错位
        sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data
    # 数据切片
    batched_data.extend(list(batch_slice(sorted_data, batch_size)))
    # 切片打乱
    if shuffle:
        np.random.shuffle(batched_data)
    # 每次调用方法给出一个迭代批次的数据
    for batch in batched_data:
        yield batch


# some function
from sklearn.metrics import f1_score, precision_score, recall_score


def get_score(y_ture, y_pred):
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_ture, y_pred, average='macro') * 100
    p = precision_score(y_ture, y_pred, average='macro') * 100
    r = recall_score(y_ture, y_pred, average='macro') * 100

    return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), reformat(f1, 2)


def reformat(num, n):
    return float(format(num, '0.' + str(n) + 'f'))


# build trainer

import time
from sklearn.metrics import classification_report

clip = 5.0
epochs = 1
early_stops = 3
log_interval = 50

test_batch_size = 128
train_batch_size = 128

save_model = './cnn.bin'
save_test = './cnn.csv'


class Trainer():
    def __init__(self, model, vocab):
        # 模型
        self.model = model
        # 是否打印训练日志
        self.report = True
        # 转化后的训练数据
        self.train_data = get_examples(train_data, vocab)
        # 一共多少批次
        self.batch_num = int(np.ceil(len(self.train_data) / float(train_batch_size)))
        # 转化后的验证数据
        self.dev_data = get_examples(dev_data, vocab)
        # 转化后的测试数据
        self.test_data = get_examples(test_data, vocab)

        # 交叉熵损失函数(由于是多分类)
        self.criterion = nn.CrossEntropyLoss()

        # 标签名称
        self.target_names = vocab.target_names

        # 优化器 优化模型中所有参数
        self.optimizer = Optimizer(model.all_parameters)

        # 相关计数/标记
        self.step = 0
        # 提前终止标识
        self.early_stop = -1
        # 最优训练集f1 最优验证集f1
        self.best_train_f1, self.best_dev_f1 = 0, 0
        # 循环训练次数
        self.last_epoch = epochs

    def train(self):
        logging.info('Start training...')
        for epoch in range(1, epochs + 1):
            train_f1 = self._train(epoch)

            dev_f1 = self._eval(epoch)

            if self.best_dev_f1 <= dev_f1:
                logging.info(
                    "Exceed history dev = %.2f, current dev = %.2f" % (self.best_dev_f1, dev_f1))
                torch.save(self.model.state_dict(), save_model)

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == early_stops:
                    logging.info(
                        "Eearly stop in epoch %d, best train: %.2f, dev: %.2f" % (
                            epoch - early_stops, self.best_train_f1, self.best_dev_f1))
                    self.last_epoch = epoch
                    break

    def test(self):
        self.model.load_state_dict(torch.load(save_model))
        self._eval(self.last_epoch + 1, test=True)

    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []
        for batch_data in data_iter(self.train_data, train_batch_size, shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            batch_outputs = self.model(batch_inputs)
            loss = self.criterion(batch_outputs, batch_labels)
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

            nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=clip)
            for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                optimizer.step()
                scheduler.step()
            self.optimizer.zero_grad()

            self.step += 1

            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time

                lrs = self.optimizer.get_lr()
                logging.info(
                    '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                        epoch, self.step, batch_idx, self.batch_num, lrs,
                        losses / log_interval,
                        elapsed / log_interval))

                losses = 0
                start_time = time.time()

            batch_idx += 1

        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time

        # reformat
        overall_losses = reformat(overall_losses, 4)
        score, f1 = get_score(y_true, y_pred)

        logging.info(
            '| epoch {:3d} | score {} | f1 {} | loss {:.4f} | time {:.2f}'.format(epoch, score, f1,
                                                                                  overall_losses,
                                                                                  during_time))
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            logging.info('\n' + report)

        return f1

    def _eval(self, epoch, test=False):
        self.model.eval()
        start_time = time.time()
        data = self.test_data if test else self.dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                batch_outputs = self.model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            score, f1 = get_score(y_true, y_pred)

            during_time = time.time() - start_time

            if test:
                df = pd.DataFrame({'label': y_pred})
                df.to_csv(save_test, index=False, sep=',')
            else:
                logging.info(
                    '| epoch {:3d} | dev | score {} | f1 {} | time {:.2f}'.format(epoch, score, f1,
                                                                                  during_time))
                if set(y_true) == set(y_pred) and self.report:
                    report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                    logging.info('\n' + report)

        return f1

    # 将数据转化为张量
    def batch2tensor(self, batch_data):
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            doc_labels.append(doc_data[0])
            doc_lens.append(doc_data[1])
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        max_doc_len = max(doc_lens)
        max_sent_len = max(doc_max_sent_len)

        batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(doc_labels)
        # [lable_id, doc长度（多少段）, doc] doc=[段落长度, 词id, 词向量]
        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                sent_data = batch_data[b][2][sent_idx]
                for word_idx in range(sent_data[0]):
                    batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                    batch_masks[b, sent_idx, word_idx] = 1

        if use_cuda:
            batch_inputs1 = batch_inputs1.to(device)
            batch_inputs2 = batch_inputs2.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)
        # 输入张量1(词id转换) 张量2(词向量转换) mask张量(全为1) 标签张量
        return (batch_inputs1, batch_inputs2, batch_masks), batch_labels


# 训练模型
trainer = Trainer(model, vocab)
trainer.train()

# 测试模型
trainer.test()
