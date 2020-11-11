import math


# 基于递归，使用使用分治法的思路，将数据一次分成两份 向下直到分为两两一对
# 分别进行排序 之后再沿着分裂方向合并 由于子序列已经有序再合并过程中只需要两个下标指向第一个位置比较就可以快速得到结果
# 时间复杂度为O(nlogn) 代价是需要更多的内存空间 典型的以空间换时间

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
print(mergeSort(arr))
