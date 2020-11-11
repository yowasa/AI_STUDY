# 首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置。
# 重复从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。
# 算法非常直观 不占用额外的内存空间
# 算法时间复杂度是O(n^2)
def selectionSort(arr):
    for i in range(len(arr) - 1):
        # 记录最小数的索引
        minIndex = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minIndex]:
                minIndex = j
        # i 不是最小数时，将 i 和最小数进行交换
        if i != minIndex:
            arr[i], arr[minIndex] = arr[minIndex], arr[i]
    return arr


arr = [22, 34, 3, 32, 82, 55, 89, 50, 37, 5, 64, 35, 9, 70]
print(selectionSort(arr))
