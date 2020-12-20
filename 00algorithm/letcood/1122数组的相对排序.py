'''
给你两个数组，arr1 和 arr2，

arr2 中的元素各不相同
arr2 中的每个元素都出现在 arr1 中
对 arr1 中的元素进行排序，使 arr1 中项的相对顺序和 arr2 中的相对顺序相同。未在 arr2 中出现过的元素需要按照升序放在 arr1 的末尾。

 

示例：

输入：arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
输出：[2,2,2,1,4,3,3,9,6,7,19]

arr1.length, arr2.length <= 1000
0 <= arr1[i], arr2[i] <= 1000
arr2 中的元素 arr2[i] 各不相同
arr2 中的每个元素 arr2[i] 都出现在 arr1 中

'''
from typing import List
import functools


# 以arr2的位置做大小的比较的依据 重写compare方法
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        map = {j: i for i, j in enumerate(arr2)}

        def cmp(a, b):
            a_ = map.get(a)
            b_ = map.get(b)
            if a_ is not None and b_ is not None:
                if a_ > b_:
                    return 1
                elif a_ < b_:
                    return -1
                else:
                    return 0
            if a_ is not None and b_ is None:
                return -1
            if a_ is None and b_ is not None:
                return 1
            if a > b:
                return 1
            elif a < b:
                return -1
            else:
                return 0

        return sorted(arr1, key=functools.cmp_to_key(cmp))

# 使用元组排序
class Solution2:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        def mycmp(x: int) -> (int, int):
            return (0, rank[x]) if x in rank else (1, x)

        rank = {x: i for i, x in enumerate(arr2)}
        arr1.sort(key=mycmp)
        return arr1


print(Solution().relativeSortArray(arr1=[28, 6, 22, 8, 44, 17], arr2=[22, 28, 8, 6]))
