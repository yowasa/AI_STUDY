from typing import List

'''
给定一个整数数组 nums，返回区间和在 [lower, upper] 之间的个数，包含 lower 和 upper。
区间和 S(i, j) 表示在 nums 中，位置从 i 到 j 的元素之和，包含 i 和 j (i ≤ j)。

说明:
最直观的算法复杂度是 O(n2) ，请在此基础上优化你的算法。

输入: nums = [-2,5,-1], lower = -2, upper = 2,
输出: 3 
解释: 3个区间分别是: [0,0], [2,2], [0,2]，它们表示的和分别为: -2, -1, 2。

'''


# 一般思路解法 双层循环嵌套 满足条件计数
class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        num_len = len(nums)
        count = 0
        for i in range(num_len):
            for j in range(i, num_len):
                sum_num = sum(nums[i:j + 1])
                if sum_num >= lower and sum_num <= upper:
                    count += 1
        return count

import bisect
# 问题转化 假设原有数组为x S(i, j)=求和：x(i)到x(j)
# 假设维护一个数组p p(i)记录的是x(0)到x(i)之和 则S(i, j)=p(j)-p(i-1)
# 简单的就是求有多少个S(i, j)满足lower到upper的条件 进行遍历时就是先求出p  lower<p(j)-p(i-1)<upper
# 对公式化简可得  p(j)-upper<p(i-1)<p(j)-lower
# i<j 我们在迭代求p的同时可以进行子集数量的统计 令当前求导的节点为j 去统计0<i<j满足p(j)-upper<p(i-1)<p(j)-lower的个数 遍历完成则求解完成
# 由于涉及到i-1 我们设p(-1)为0 方便计算
# 维护一个递增t数列用于储存p 在p(j)加入数列后 增加的子区间个数变为t中有一个连续的范围在p(j)-upper和p(j)-lower之间的个数（下标之差）
# 使用bisect可以很方便的求解
class Solution2:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        res, pre, now = 0, [0], 0
        for n in nums:
            now += n
            res += bisect.bisect_right(pre, now - lower) - bisect.bisect_left(pre, now - upper)
            bisect.insort(pre, now)
        return res



print(Solution2().countRangeSum([-2, 5, -1], -2, 2))
