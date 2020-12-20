'''
给你一个字符串 s ，请你根据下面的算法重新构造字符串：

从 s 中选出 最小 的字符，将它 接在 结果字符串的后面。
从 s 剩余字符中选出 最小 的字符，且该字符比上一个添加的字符大，将它 接在 结果字符串后面。
重复步骤 2 ，直到你没法从 s 中选择字符。
从 s 中选出 最大 的字符，将它 接在 结果字符串的后面。
从 s 剩余字符中选出 最大 的字符，且该字符比上一个添加的字符小，将它 接在 结果字符串后面。
重复步骤 5 ，直到你没法从 s 中选择字符。
重复步骤 1 到 6 ，直到 s 中所有字符都已经被选过。
在任何一步中，如果最小或者最大字符不止一个 ，你可以选择其中任意一个，并将其添加到结果字符串。

请你返回将 s 中字符重新排序后的 结果字符串 。

 

示例 1：
输入：s = "aaaabbbbcccc"
输出："abccbaabccba"
解释：第一轮的步骤 1，2，3 后，结果字符串为 result = "abc"
第一轮的步骤 4，5，6 后，结果字符串为 result = "abccba"
第一轮结束，现在 s = "aabbcc" ，我们再次回到步骤 1
第二轮的步骤 1，2，3 后，结果字符串为 result = "abccbaabc"
第二轮的步骤 4，5，6 后，结果字符串为 result = "abccbaabccba"
示例 2：
输入：s = "rat"
输出："art"
解释：单词 "rat" 在上述算法重排序以后变成 "art"
示例 3：
输入：s = "leetcode"
输出："cdelotee"
示例 4：
输入：s = "ggggggg"
输出："ggggggg"
示例 5：
输入：s = "spo"
输出："ops"


提示：
1 <= s.length <= 500
s 只包含小写英文字母。
'''


# 传统思路 扫描转移
class Solution:
    def sortString(self, s: str) -> str:
        target = list(s)
        length = len(target)
        target.sort()
        flag = True
        check_index = 0
        put_index = 0
        befor = '`'
        while put_index != length - 1:
            if flag:
                if target[check_index] > befor:
                    tmp = target.pop(check_index)
                    befor = tmp
                    target.insert(put_index, tmp)
                    put_index += 1
                if check_index == length - 1:
                    flag = not flag
                    befor = '{'
                else:
                    check_index += 1
            else:
                if target[check_index] < befor:
                    tmp = target.pop(check_index)
                    befor = tmp
                    target.insert(put_index, tmp)
                    put_index += 1
                else:
                    check_index -= 1
                if check_index == put_index - 1:
                    flag = not flag
                    befor = '`'
                    check_index += 1
        return "".join(target)


# 使用计数桶排 然后统一转化
class Solution2:
    def sortString(self, s: str) -> str:
        num = [0] * 26
        for ch in s:
            num[ord(ch) - ord('a')] += 1

        ret = list()
        while len(ret) < len(s):
            for i in range(26):
                if num[i]:
                    ret.append(chr(i + ord('a')))
                    num[i] -= 1
            for i in range(25, -1, -1):
                if num[i]:
                    ret.append(chr(i + ord('a')))
                    num[i] -= 1

        return "".join(ret)

import collections
# 非传统桶排
class Solution3:
    def sortString(self, s: str) -> str:
        t, result = collections.Counter(s), ""
        sort = sorted(t)
        reverse_sort = list(reversed(sort))
        while len(result) < len(s):
            for k in sort:
                if t[k]:
                    result, t[k] = result + k, t[k] - 1
            for k in reverse_sort:
                if t[k]:
                    result, t[k] = result + k, t[k] - 1
        return result



print(Solution().sortString("aaaabbbbcccc"))
