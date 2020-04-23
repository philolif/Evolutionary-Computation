import numpy as np


class SA:
    """
    模拟退火算法：寻找函数的最小值
    """

    def __init__(self, current_temp, gamma, final_temp, x0, st_x, func):
        # 初始温度
        self.current_temp = current_temp
        # 温度采用线性递减策略
        self.gamma = gamma
        # 终止温度
        self.final_temp = final_temp
        # 空间维数
        self.n = x0.size
        # 约束x坐标的范围
        self.st_x = st_x
        # 目标函数
        self.func = func
        # 当前找到的解
        self.solution = x0
        # 当前解对应的函数值
        self.value = func(x0)
        # 历史最优解
        self.best_solution = x0
        self.best_value = func(x0)

    def solve(self):
        while self.current_temp > self.final_temp:
            accept = False
            offset = self.get_offset()
            next_solution = self.solution + offset
            next_solution[next_solution > self.st_x[1]] = self.st_x[1]
            next_solution[next_solution < self.st_x[0]] = self.st_x[0]
            if self.func(next_solution) < self.value:
                accept = True
            elif np.random.rand() < np.exp((self.value - self.func(next_solution)) / self.current_temp):
                accept = True
            if accept:
                self.solution = next_solution
                self.value = self.func(next_solution)
                if self.value < self.best_value:
                    self.best_solution = self.solution
                    self.best_value = self.value
            self.current_temp -= self.gamma
        return self.best_solution, self.best_value

    def get_offset(self):
        # 产生随机偏移量，和温度相关
        v = np.random.rand(self.n) - 0.5
        v *= self.current_temp / 100
        return v


def ackley(x):
    return - 20 * np.exp(-0.2 * np.sqrt((x * x).mean())) - np.exp(np.cos(2 * np.pi * x).mean()) + 20 + np.exp(1)


if __name__ == '__main__':
    # ackley函数的最小值
    for i in range(10):
        sa = SA(
            current_temp=500,
            gamma=0.1,
            final_temp=1,
            st_x=np.array([-10, 10]),
            x0=np.random.randn(2),
            func=ackley
        )
        solution, value = sa.solve()
        print(solution, value)
