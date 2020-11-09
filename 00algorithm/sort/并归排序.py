import math
import datetime


# 排序方法
def mergeSort(arr):
    if (len(arr) < 2):
        return arr
    # 将数据分成两份
    middle = math.floor(len(arr) / 2)
    left, right = arr[0:middle], arr[middle:]
    # 无线下分 合并左右
    return merge(mergeSort(left), mergeSort(right))


# 合并左右
def merge(left, right):
    result = []
    # 当左右都有的时候 判断左右大小 逐个取出 比较大小放入结果集
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0));
    while left:
        result.append(left.pop(0))
    while right:
        result.append(right.pop(0));
    return result


arr = [22, 34, 3, 32, 82, 55, 89, 50, 37, 5, 64, 35, 9, 70]
begin = datetime.datetime.now()
print(mergeSort(arr))
end  = datetime.datetime.now()
print(f'系统耗时{end-begin}')
