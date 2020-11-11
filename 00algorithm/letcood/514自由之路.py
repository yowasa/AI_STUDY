'''
视频游戏“辐射4”中，任务“通向自由”要求玩家到达名为“Freedom Trail Ring”的金属表盘，并使用表盘拼写特定关键词才能开门。

给定一个字符串 ring，表示刻在外环上的编码；给定另一个字符串 key，表示需要拼写的关键词。您需要算出能够拼写关键词中所有字符的最少步数。

最初，ring 的第一个字符与12:00方向对齐。您需要顺时针或逆时针旋转 ring 以使 key 的一个字符在 12:00 方向对齐，然后按下中心按钮，以此逐个拼写完 key 中的所有字符。

旋转 ring 拼出 key 字符 key[i] 的阶段中：

您可以将 ring 顺时针或逆时针旋转一个位置，计为1步。旋转的最终目的是将字符串 ring 的一个字符与 12:00 方向对齐，并且这个字符必须等于字符 key[i] 。
如果字符 key[i] 已经对齐到12:00方向，您需要按下中心按钮进行拼写，这也将算作 1 步。按完之后，您可以开始拼写 key 的下一个字符（下一阶段）, 直至完成所有拼写。

提示：

ring 和 key 的字符串长度取值范围均为 1 至 100；
两个字符串中都只有小写字符，并且均可能存在重复字符；
字符串 key 一定可以由字符串 ring 旋转拼出。
'''


# 转化为路径规划问题
# 先将所有字符对应位置的下标存下来
# 求最小路径 使用深度优先遍历即可
class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        import collections, functools
        lookup = collections.defaultdict(list)
        for i in range(len(ring)):
            lookup[ring[i]].append(i)

        # 增加了cache防止重复计算问题
        @functools.lru_cache(None)
        def dfs(cur, k):
            if k == len(key):
                return 0
            # 需要的步骤
            res = float("inf")
            for j in lookup[key[k]]:
                # 当前位置到需要寻找的下标位置的距离
                tmp = abs(cur - j)
                # 由于是环形的所有有两种转法 求最小的那种 则tmp为旋转了多少步
                tmp = min(tmp, len(ring) - tmp)
                # 假设采用了当前的转法 把当前位置转到该位置 +1 是按button所需要的步骤 然后假设按当前转发看转下一个需要的最小步骤
                # 如果使用下一个下标更好则覆盖 否则采用上一个步骤
                res = min(res, min(tmp, len(ring) - tmp) + 1 + dfs(j, k + 1))
            return res

        return dfs(0, 0)


# 类Viterbi算法
class Solution2:
    def findRotateSteps(self, ring: str, key: str) -> int:
        ring_length = len(ring)

        def locate(k_):
            # ring 中有多少那一位 key，返回下标
            return [i for i in range(ring_length) if ring[i] == k_]

        def rotate(begin, end) -> int:
            # 计算正转和反转哪一个更近
            max_ = max((begin, end))
            min_ = min((begin, end))
            return min((max_ - min_, min_ - max_ + ring_length))

        # 初始化
        candidates = [(0, 0)]

        for k in key:
            k_indices = locate(k)
            new_candidates = []
            for index in k_indices:
                # 对于上一个 key 的所有可能到下一个 key 的最短步骤（dp）
                steps = min([rotate(c[0], index) + c[1] for c in candidates])
                new_candidates.append((index, steps))
            candidates = new_candidates

        return len(key) + min([c[1] for c in candidates])


print(Solution2().findRotateSteps(ring="godding", key="gd"))
