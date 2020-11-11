import math


# 堆排序的平均时间复杂度为 Ο(nlogn)

# 定义堆结构 假设使用最大堆 则每个父节点都大于其子节点的值
# 在数组中模拟堆的树形结构 则为 取一半长度的节点为父节点 其对应的子节点分别为父节点下标的2 * x + 1 和 2 * i + 2 很明显0位置就是堆顶
# 先构建最大堆
def buildMaxHeap(arr):
    for i in range(math.floor(len(arr) / 2), -1, -1):
        heapify(arr, i)


# 过程为从最下层父节点开始 依次比较左右节点与父节点关系
# 发现左右节点比父节点大的情况就进行替换
# 当执行完最下层之后 有可能会父节点与父节点进行替换 导致子节点树需要重新比较 故需要一个递归式
# 全部执行完之后就得到了一个最大堆结构的树
def heapify(arr, i):
    left = 2 * i + 1
    right = 2 * i + 2
    largest = i
    if left < arrLen and arr[left] > arr[largest]:
        largest = left
    if right < arrLen and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        swap(arr, i, largest)
        heapify(arr, largest)


def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


# 构建完成最大堆
# 0位置肯定是最大的数
# 倒叙遍历所有数据 让0与其互换
# 每一次让长度-1 把最大值锁在末尾
# 当锁完所有的值排序也就完成了
def heapSort(arr):
    global arrLen
    arrLen = len(arr)
    buildMaxHeap(arr)
    for i in range(len(arr) - 1, 0, -1):
        swap(arr, 0, i)
        arrLen -= 1
        heapify(arr, 0)
    return arr


arr = [22, 34, 3, 32, 82, 55, 89, 50, 37, 5, 64, 35, 9, 70]
print(heapSort(arr))
