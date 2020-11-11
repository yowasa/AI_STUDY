import math


# 也称递减增量排序算法 是基于插入排序优化的非稳定排序算法
# 定义一个递减到1的序列 称之为希尔序列 最大值大于需要排序的数组长度 n1,n2...1
# 以gap为n1对数据进行分组 对每一组数据进行一次插入排序
# 再以gap为n2对数据进行分组排序 直到分组为1为止
# 由于经过n1,n2...的优化 最后排序时整体已经比较有序 基本不用动太多
# 正确的希尔序列可以使时间复杂度到O(n^(3/2))


def shellSort(arr):
    gap = 1
    while (gap < len(arr) / 3):
        gap = gap * 3 + 1
    while gap > 0:
        for i in range(gap, len(arr)):
            temp = arr[i]
            j = i - gap
            while j >= 0 and arr[j] > temp:
                arr[j + gap] = arr[j]
                j -= gap
            arr[j + gap] = temp
        gap = math.floor(gap / 3)
    return arr


arr = [22, 34, 3, 32, 82, 55, 89, 50, 37, 5, 64, 35, 9, 70]
print(shellSort(arr))
