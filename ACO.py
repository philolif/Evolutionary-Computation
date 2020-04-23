import numpy as np


class Ant:
    def __init__(self, n):
        # 城市数目
        self.n = n

    def init(self):
        # 未访问过的城市:0,1,2, ..., n-1
        self.allowed = [i for i in range(self.n)]
        # 随机初始化蚂蚁的初始城市
        city = np.random.randint(self.n)
        self.path = [city]
        self.allowed.remove(city)
        # 路径长度
        self.distance = 0
        # 信息素更新矩阵 pher[i,j]==1表示蚂蚁经过了i->j这条路径
        self.pher = np.zeros((self.n, self.n))

    def get_pheromone(self, Q):
        # 每只蚂蚁携带Q个单位的信息素
        return Q / self.distance * self.pher

    def get_path(self):
        return self.path, self.distance

    def travel(self, D, P, alpha, beta):
        while self.allowed:
            # 计算访问下一个城市的概率
            prob = []
            for city in self.allowed:
                d = D[self.path[-1], city]
                pher = P[self.path[-1], city]
                prob.append(
                    ((1 / d) ** alpha) * (pher ** beta)
                )
            # 归一化
            prob = np.array(prob) / sum(prob)
            # 选择下一个城市
            next_city = np.random.choice(self.allowed, p=prob)
            self.distance += D[self.path[-1], next_city]
            # 如果i->j和j->i的距离是不对称的，那么更新方式为：self.pher[self.path[-1], next_city] = 1.0
            self.pher[[self.path[-1], next_city], [next_city, self.path[-1]]] = 1.0
            self.path.append(next_city)
            self.allowed.remove(next_city)
        # 回到起点
        self.distance += D[self.path[-1], self.path[0]]
        self.path.append(self.path[0])
        # 如果i->j和j->i的距离是不对称的，那么更新方式为：self.pher[self.path[-1], self.path[0]] = 1.0
        self.pher[[self.path[-1], self.path[0]], [self.path[0], self.path[-1]]] = 1.0

class TSP:
    def __init__(self, m, n, D, P, Q, alpha, beta, rho, num_iter):
        # 蚂蚁个数
        self.m = m
        # 城市个数
        self.n = n
        # 距离矩阵
        self.D = D
        # 信息素矩阵
        self.P = P
        # 当前最佳路径
        self.path = None
        self.distance = np.inf
        # 每只蚂蚁携带的信息素
        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        # 信息素蒸发系数
        self.rho = rho
        # 迭代次数
        self.num_iter = num_iter
        # 蚂蚁种群
        self.ants = [Ant(n) for _ in range(m)]

    def solve(self):
        for _ in range(self.num_iter):
            phers = np.zeros((self.n, self.n))
            for ant in self.ants:
                ant.init()
            for ant in self.ants:
                ant.travel(self.D, self.P, self.alpha, self.beta)
                phers += ant.get_pheromone(self.Q)
                path, dist = ant.get_path()
                if dist < self.distance:
                    self.path = path
                    self.distance = dist
            # 更新信息素
            self.P = (1 - self.rho) * self.P + phers
        return self.path, self.distance


if __name__ == '__main__':
    # 4个城市对称TSP问题
    D = np.array([
        [0, 11, 2, 1],
        [11, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ])
    # 为了简单起见，路径上的信息素初始化为0
    P = np.ones((4, 4))
    tsp = TSP(
        m=5,
        n=4,
        D=D,
        P=P,
        Q=1,
        alpha=0.5,
        beta=0.5,
        rho=0.5,
        num_iter=5
    )
    path, length = tsp.solve()
    print(path, length)
