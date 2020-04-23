import numpy as np
import operator


class Solution:
    __slots__ = ('value', 'cost')

    def __init__(self, value, cost):
        self.value = value
        self.cost = cost


class Item:
    __slots__ = ('solution', 'nTimes')

    def __init__(self, solution, nTimes):
        self.solution = solution
        self.nTimes = nTimes


class TS:
    def __init__(self, nCities, nTimes, num_iter, dist):
        self.nCities = nCities
        self.nTimes = nTimes
        self.num_iter = num_iter
        self.dist = dist
        self.best = None
        self.table = []

    def getRandomSolution(self):
        temp = [i for i in range(self.nCities)]
        np.random.shuffle(temp)
        return Solution(temp, self.getCost(temp))

    def getCost(self, x):
        cost = 0
        for i in range(self.nCities):
            cost += self.dist[x[i], x[(i + 1) % self.nCities]]
        return cost

    def get_candidates(self, x):
        cans = []
        for i in range(self.nCities - 1):
            temp = x.value[:]
            temp[i], temp[i + 1] = temp[i + 1], temp[i]
            cans.append(Solution(temp, self.getCost(temp)))
        return cans

    def update_table(self):
        delete = []
        for item in self.table:
            item.nTimes -= 1
            if item.nTimes == 0:
                delete.append(item)
        for item in delete:
            self.table.remove(item)

    def isAvailable(self, x):
        for item in self.table:
            if operator.eq(item.solution.value, x.value):
                return False
        return True

    def solve(self):
        # 初始化
        x = self.getRandomSolution()
        self.best = x
        self.table.append(Item(x, self.nTimes))
        # 开始搜索
        for _ in range(self.num_iter):
            self.update_table()
            cans = self.get_candidates(x)
            while cans:
                x = min(cans, key=lambda s: s.cost)
                if self.isAvailable(x):
                    self.table.append(Item(x, self.nTimes))
                    break
                else:
                    cans.remove(x)
            else:
                # 特赦
                x = min(self.table, key=lambda i: i.solution.cost).solution
            if x.cost < self.best.cost:
                self.best = x
        return self.best


if __name__ == '__main__':
    dist = np.array([
        [0, 1, 5, 4, 3],
        [3, 0, 2, 6, 1],
        [5, 2, 0, 1, 4],
        [1, 6, 3, 0, 3],
        [3, 3, 1, 4, 0]
    ])
    ts = TS(
        nCities=5,
        nTimes=3,
        num_iter=10,
        dist=dist
    )
    s = ts.solve()
    print(s.value)
    print(s.cost)
