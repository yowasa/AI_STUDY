'''
给定一个以字符串表示的非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。

注意:

num 的长度小于 10002 且 ≥ k。
num 不会包含任何前导零。

示例 1 :

输入: num = "1432219", k = 3
输出: "1219"
解释: 移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219。
示例 2 :

输入: num = "10200", k = 1
输出: "200"
解释: 移掉首位的 1 剩下的数字为 200. 注意输出不能有任何前导零。
示例 3 :

输入: num = "10", k = 2
输出: "0"
解释: 从原数字移除所有的数字，剩余为空就是0。
'''


# 由于str不能直接进行删除操作 故需要进行转化 先找出需要删除的位置再删除
# 每一次取剩余个数+1个数字 找到最小的 最小的之前全部标记为待删除 并记录标记的个数 loop直到找齐
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        num = list(num)
        step = 0
        index = 0
        drop_index = []
        while step < k:
            clusor = index
            for i in range(index, index + k - step + 1):
                if i > len(num) - 1:
                    clusor = i
                    break
                if int(num[clusor]) > int(num[i]):
                    clusor = i
            drop_index.extend(range(index, clusor))
            step += clusor - index
            index = clusor + 1
        cnt = 0
        for a in drop_index:
            del num[a - cnt]
            cnt += 1
        while len(num) > 0 and int(num[0]) == 0:
            del num[0]
        if len(num) == 0:
            return '0'
        return "".join(num)


# 使用栈优化
class Solution2:
    def removeKdigits(self, num: str, k: int) -> str:

        stack = []
        step = 0
        for key, value in enumerate(num):
            if not stack:
                stack.append(value)
                continue
            index = key - step-1
            int_value = int(value)
            while index >= 0 and int_value < int(stack[index]) and step < k:
                step += 1
                index -= 1
                stack.pop(-1)
            stack.append(value)
        if step < k:
            while step < k:
                stack.pop(-1)
                step += 1
        while len(stack) > 0 and int(stack[0]) == 0:
             stack.pop(0)
        if len(stack) == 0:
            return '0'
        return "".join(stack)


print(Solution2().removeKdigits(num="10", k=2))
