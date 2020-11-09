class Solution(object):
    def countPrimes(self, n):
        if n < 3:
            return 0
        d = [1] * n
        d[0] = d[1] = 0
        for i in range(2, int(n * 0.5) + 1):
            if d[i] == 1:
                for j in range(i * i, n, i):
                    d[j] = 0
        return sum(d)


import time

begin = time.time()
print(Solution().countPrimes(66666))
end = time.time()
print(str(end - begin) + "秒")
