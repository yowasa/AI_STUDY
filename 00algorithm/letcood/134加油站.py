'''
在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

说明: 

如果题目有解，该答案即为唯一答案。
输入数组均为非空数组，且长度相同。
输入数组中的元素均为非负数。
'''
from typing import List


# 广优搜索
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        length = len(gas)
        ccost = list(map(lambda x: x[0] - x[1], zip(gas, cost)))
        gas_left = [0] * len(gas)
        for i in range(length):
            for j in range(length):
                next = i + j if i + j < length else i + j - length
                if gas_left[i] == 'naf':
                    continue
                value = gas_left[i] + ccost[next]
                if value < 0:
                    gas_left[i] = 'naf'
                else:
                    gas_left[i] = value
        for index, value in enumerate(gas_left):
            if value != 'naf':
                return index
        return -1


import collections


class Solution2:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        length = len(gas)
        ccost = list(map(lambda x: x[0] - x[1], zip(gas, cost)))
        que = collections.deque(list(range(length)))
        gas_left = {i: 0 for i in que}
        i = 0
        flag = -1
        while i < length and que:
            x = que.popleft()
            if x <= flag:
                i += 1
            flag = x
            next = i + x if i + x < length else i + x - length
            value = gas_left[x] + ccost[next]
            if value >= 0:
                gas_left[x] = value
                que.append(x)
        if not que:
            return -1
        return que.popleft()


class Solution3:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        start = fuel = 0
        for i in range(len(gas)):
            if fuel + gas[i] - cost[i] >= 0:
                fuel += gas[i] - cost[i]
            else:
                fuel = 0
                start = i + 1
        for i in range(start):
            if fuel + gas[i] - cost[i] >= 0:
                fuel += gas[i] - cost[i]
            else:
                return -1
        return start


print(Solution2().canCompleteCircuit(gas=[1, 2, 3, 4, 5], cost=[3, 4, 5, 1, 2]))
