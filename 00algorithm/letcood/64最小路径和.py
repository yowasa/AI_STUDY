# 给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
# 说明：每次只能向下或者向右移动一步。

class Solution:
    def minPathSum(self, grid) -> int:
        high = len(grid)
        weight = len(grid[0])
        dp = []
        for i in range(high):
            dp.append([0] * weight)

        for i in range(high):
            for j in range(weight):
                if i == 0:
                    if j == 0:
                        dp[i][j] = grid[i][j]
                    else:
                        dp[i][j] = grid[i][j] + dp[i][j - 1]
                else:
                    if j == 0:
                        dp[i][j] = grid[i][j] + dp[i - 1][j]
                    else:
                        if dp[i][j - 1] <= dp[i - 1][j]:
                            dp[i][j] = dp[i][j - 1] + grid[i][j]
                        else:
                            dp[i][j] = dp[i - 1][j] + grid[i][j]
        return dp[high - 1][weight - 1]


test = [[1, 4, 8, 6, 2, 2, 1, 7], [4, 7, 3, 1, 4, 5, 5, 1], [8, 8, 2, 1, 1, 8, 0, 1], [8, 9, 2, 9, 8, 0, 8, 9],
        [5, 7, 5, 7, 1, 8, 5, 5], [7, 0, 9, 4, 5, 6, 5, 6], [4, 9, 9, 7, 9, 1, 9, 0]]
import time

begin = time.time()
print(Solution().minPathSum(test))
end = time.time()
print(str(end - begin) + "秒")
