import numpy as np


# v(k+1) = w*v(k) + c1*r1*(gbest-x) + c2*r2*(pbest-x)
# x(k+1) = x(k) + v(k+1)


class Particle:
    """
    st_x：位置约束，粒子每个维度上的坐标范围必须处于[st_x[0], st_x[1]]之间
    st_v：速度约束，粒子每个维度上的速度范围必须处于[st_v[0], st_v[1]]之间
    position：粒子的当前位置
    velocity：粒子的当前速度
    pbest：粒子自身历史记录的最佳位置
    pvalue：粒子自身历史记录的最佳值
    """
    st_x = None
    st_v = None

    def __init__(self, x, v, pbest, pvalue):
        self.position = x
        self.velocity = v
        self.pbest = pbest
        self.pvalue = pvalue

    def update_velocity(self, v):
        v[v < self.st_v[0]] = self.st_v[0]
        v[v > self.st_v[1]] = self.st_v[1]
        self.velocity = v

    def update_position(self, func):
        # 更新粒子自身的位置，以及判断是否要更新pbest
        self.position += self.velocity
        self.position[self.position < self.st_x[0]] = self.st_x[0]
        self.position[self.position > self.st_x[1]] = self.st_x[1]
        if self.pvalue > func(self.position):
            self.pvalue = func(self.position)
            self.pbest = self.position.copy()


class PSO:
    """
    gbest：粒子群历史记录的最佳位置
    gvalue：粒子群历史记录的最佳值
    """
    gbest = None
    gvalue = np.inf

    def __init__(self, n_dims, n_particles, st_x, st_v, w, c1, c2, num_iter, func):
        # 初始化空间维度
        self.n_dims = n_dims
        # 初始化粒子群数目
        self.n_particles = n_particles
        # 目标函数
        self.func = func
        # 粒子惯性权重
        self.w = w
        # 全局部分学习率
        self.c1 = c1
        # 自我认知部分学习率
        self.c2 = c2
        # 迭代次数
        self.num_iter = num_iter
        # 存放粒子群的列表（容器）
        self.particles = []
        # 初始化粒子的位置和速度约束
        Particle.st_x = st_x
        Particle.st_v = st_v
        # 初始化粒子群
        for _ in range(n_particles):
            # 初始化粒子的随机位置在 st_x[0]~st_x[1] 
            x = (st_x[1] - st_x[0]) * np.random.rand(n_dims) + st_x[0]
            # 计算当前评估值
            pvalue = func(x)
            # 初始化一个粒子
            self.particles.append(
                Particle(
                    x=x,
                    v=(st_v[1] - st_v[0]) * np.random.rand(n_dims) + st_v[0],
                    pbest=x.copy(),
                    pvalue=pvalue
                )
            )

            if self.gvalue > pvalue:
                self.gvalue = pvalue
                self.gbest = x.copy()

    def solve(self):
        # 开始迭代
        for index in range(1, self.num_iter + 1):
            for particle in self.particles:
                v = self.w * particle.velocity + self.c1 * np.random.rand() * (self.gbest - particle.position) + \
                    self.c2 * np.random.rand() * (particle.pbest - particle.position)
                particle.update_velocity(v)
                particle.update_position(self.func)
            for particle in self.particles:
                if particle.pvalue < self.gvalue:
                    self.gvalue = particle.pvalue
                    self.gbest = particle.pbest.copy()
        return self.gbest, self.gvalue


def ackley(x):
    return - 20 * np.exp(-0.2 * np.sqrt((x * x).mean())) - np.exp(np.cos(2 * np.pi * x).mean()) + 20 + np.exp(1)


if __name__ == "__main__":
    # 测试不同迭代次数搜索出来的最优解情况
    for i in range(5):
        pso = PSO(
            n_dims=3,
            n_particles=20,
            st_x=(-10, 10),
            st_v=(-0.2, 0.2),
            w=0.8,
            c1=2,
            c2=2,
            num_iter=50 * i,
            func=ackley
        )
        gbest, gvalue = pso.solve()
        print("number of interations:%d\tgvalue:%f" % (50 * i, gvalue))
