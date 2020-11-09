from typing import List

'''
给出一个无重叠的，按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
输出：[[1,5],[6,9]]

输入：intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
输出：[[1,2],[3,10],[12,16]]
解释：这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。

'''

#思路 使用二分法左右查找对应index节点 之后对比大小进行列表替换操作


class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        L, R = newInterval
        # 二分查找（左），参考bisect.bisect_left
        pL, r = 0, len(intervals)
        while (pL < r):
            m = (pL + r) // 2
            if intervals[m][1] < L:
                pL = m + 1
            else:
                r = m
        # 二分查找（右），参考bisect.bisect_right
        pR, r = pL, len(intervals)
        while (pR < r):
            m = (pR + r) // 2
            if intervals[m][0] > R:
                r = m
            else:
                pR = m + 1

        # python slice替换，待替换片段和原片段等长时，时间复杂度为O(R-L), 否则为O(N-L)
        intervals[pL:pR] = [newInterval] if pL >= pR else [[min(L, intervals[pL][0]), max(R, intervals[pR - 1][1])]]
        return intervals


print(Solution().insert(intervals=[[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], newInterval=[4, 8]))
