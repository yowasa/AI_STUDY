# 重复地走访过要排序的数列，一次比较两个元素，如果他们的顺序错误就把他们交换过来。
# 这个算法的名字由来是因为越小的元素会经由交换慢慢"浮"到数列的顶端。
# 算法时间复杂度是O(n^2)
def bubbleSort(arr):
    for i in range(1, len(arr)):
        for j in range(0, len(arr) - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


arr = [22, 34, 3, 32, 82, 55, 89, 50, 37, 5, 64, 35, 9, 70]
print(bubbleSort(arr))
