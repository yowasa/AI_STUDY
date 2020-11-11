from typing import List

'''
现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须原地修改，只允许使用额外常数空间。

以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1
'''


# 字典序列是手写简单描述很难的序列 是按照一定顺序将数据所有的排列组合全部列出来
# 例如 123的字典序列是 0:123 1:132 2:213 3:231 4:312 5:321 总结就是尽可能让左侧变化最缓慢的序列
# 经分析可得 下一个序列的数一定比当前序列的大 而且每次增大是尽可能小的增大
# 故得到算法为 下一序列的排序为:当前序列 从后向前扫描 记录最大的大数 当发现前面有比最大的大数小的时候 向后找到比他大的最小的数 进行交换
# 交换过后交换点后的数据重置为升序 以确保其增加的最小
# 若扫描全部的数发现没有比最后一个数更小的存在 则证明序列时最后一个 只需要全部倒序即可

class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        index = 0
        for i in range(len(nums) - 1, 0, -1):
            if nums[i] > nums[i - 1]:
                index = i
                break
        if index == len(nums) - 1:
            nums[index], nums[index - 1] = nums[index - 1], nums[index]
        elif index == 0:
            self.Reverse(nums, 0, len(nums) - 1)
        else:
            self.Reverse(nums, index, len(nums) - 1)
            for s in range(index, len(nums)):
                if nums[s] > nums[index - 1]:
                    nums[index - 1], nums[s] = nums[s], nums[index - 1]
                    break

    def Reverse(self, nums, start, end):
        mid = (start + end + 1) // 2
        k = 0
        for j in range(start, mid):
            nums[j], nums[end - k] = nums[end - k], nums[j]
            k += 1


# 使用二分法进行最后的查找优化
class Solution2:
    def nextPermutation(self, nums: List[int]) -> None:
        for i in range(len(nums) - 1, 0, -1):
            # 如果前面的某一个数大于最后一个数
            if nums[i - 1] >= nums[-1]:
                # 将该数放到最后 其他的数顺序平移
                nums[:] = nums[:i - 1] + nums[i:] + [nums[i - 1]]
            else:
                # 如果该数小于最后一个数 即小于最大的数 证明需要被替换
                l = i
                r = len(nums) - 1
                # 二分查找模板查找最后一个大于目标值的数
                while l < r:
                    m = l + (r - l) // 2
                    if nums[m] <= nums[i - 1]:
                        l = m + 1
                    else:
                        r = m
                tmp = nums[i - 1]
                nums[i - 1] = nums[l]
                nums[l] = tmp
                break


import bisect


# 使用bisect进行查找
class Solution2:
    def nextPermutation(self, nums: List[int]) -> None:
        index = len(nums) - 1
        # 找到从右向左最大的数（开始衰减的前一个）
        while index > 0:
            if nums[index] > nums[index - 1]:
                break
            index -= 1
        # 如果不是第一个
        if index > 0:
            # 将找到的数后面排序
            nums[index:] = sorted(nums[index:])
            # 搜索最大值的前一个数应当插入到右侧切片的什么位置
            swap_index = bisect.bisect(nums, nums[index - 1], lo=index)
            # 交换数据
            nums[swap_index], nums[index - 1] = nums[index - 1], nums[swap_index]
        # 如果是第一个直接反转
        else:
            nums.reverse()


arr = [1, 3, 2]
Solution2().nextPermutation(arr)
print(arr)
