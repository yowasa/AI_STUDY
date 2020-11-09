from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = datasets.load_digits()

# 图片转化为数组后 64
print(digits.data.shape)
# 图片8*8
print(digits.images.shape)
# y值
print(digits.target.shape)
# 随便找个图形显示出来看看
# idx=100
# print(digits.images[idx])
# plt.imshow(digits.images[idx], cmap=plt.cm.gray_r)
# plt.title(digits.target[idx])
# plt.show()

X = digits.data
Y = digits.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

algo = SVC()

algo.fit(X_train, Y_train)
train_predict = algo.predict(X_train)
test_predict = algo.predict(X_test)

print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
print(f'分类评估报告（训练集）:{classification_report(Y_train, train_predict)}')
print(f'分类评估报告（测试集）:{classification_report(Y_test, test_predict)}')
