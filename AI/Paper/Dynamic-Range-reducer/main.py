import numpy as np
import heapq
import math
import itertools
import networkx as nx
from typing import Tuple, List

class QUBODynamicRangeReducer:
    def __init__(self, Q: np.ndarray, T: int = 5, roll_depth: int = 2, 
                 branch_strategy: str = 'IMPACT', verbose: bool = False):
        """
        初始化QUBO动态范围缩减器
        
        Args:
            Q (np.ndarray): 原始QUBO矩阵(上三角形式)
            T (int): 最大搜索步数，默认5
            roll_depth (int): 策略推演深度，默认2
            branch_strategy (str): 分支策略('ALL'或'IMPACT')，默认'IMPACT'
            verbose (bool): 是否打印详细过程，默认False

        """
        self.original_Q = Q.copy()
        self.n = Q.shape[0]
        self.T = T
        self.roll_depth = roll_depth
        self.branch_strategy = branch_strategy
        self.verbose = verbose

        self.original_DR = self.dynamic_range(Q)
        
        # 状态空间
        self.best_DR = float('inf')
        self.best_state = None
        self.nodes_explored = 0
        self.nodes_pruned = 0

    def dynamic_range(self, Q: np.ndarray) -> float:
        """
        计算QUBO矩阵的动态范围(DR)
        
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            
        Returns:
            float: 计算得到的动态范围值

        """
        # 提取上三角元素
        values = []
        for i in range(self.n):
            for j in range(i, self.n):
                values.append(Q[i, j])
        
        # 计算所有非零差值
        unique_vals = sorted(set(values))
        if len(unique_vals) < 2:
            return 0.0
        
        min_diff = float('inf')
        for i in range(1, len(unique_vals)):
            diff = unique_vals[i] - unique_vals[i-1]
            if diff > 0:
                min_diff = min(min_diff, diff)
        
        max_diff = unique_vals[-1] - unique_vals[0]
        return math.log2(max_diff / min_diff) if min_diff > 0 else float('inf')

    def get_possible_actions(self, Q: np.ndarray) -> List[Tuple[int, int]]:
        """
        获取可能的动作(索引对)
        
        Args:
            Q (np.ndarray): 当前QUBO矩阵
            
        Returns:
           action(List[Tuple[int, int]]): 可能的动作索引对列表

        """
        if self.branch_strategy == 'ALL':
            return [(i, j) for i in range(self.n) for j in range(i, self.n)]
        
        # IMPACT策略
        values = []
        for i in range(self.n):
            for j in range(i, self.n):
                values.append((Q[i, j], (i, j)))
        
        # 排序，最大值最小值和最小差值对
        sorted_vals = sorted(values, key=lambda x: x[0])
        min_val, max_val = sorted_vals[0], sorted_vals[-1]
        
        actions = set()
        actions.add(min_val[1])
        actions.add(max_val[1])
        
        min_diff = float('inf')
        min_diff_pair = None
        for i in range(1, len(sorted_vals)):
            diff = sorted_vals[i][0] - sorted_vals[i-1][0]
            if diff > 0 and diff < min_diff:
                min_diff = diff
                min_diff_pair = (sorted_vals[i-1][1], sorted_vals[i][1])
        
        if min_diff_pair:
            actions.add(min_diff_pair[0])
            actions.add(min_diff_pair[1])
        
        return list(actions)

    def lb_roof_dual(self, Q: np.ndarray, fixed_vars: dict = None, exact_limit: int = 4) -> float:
        """
        使用Roof Dual下界计算QUBO矩阵.
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            fixed_vars (dict): 已固定变量的字典，键为变量索引，值为固定值
            exact_limit (int): 精确计算阈值，默认4
            
        Returns:
            lower_energy(float): 计算得到的Roof Dual下界

        """
        # clamp 固定变量
        def clamp_qubo(self, Q: np.ndarray, fixed_vars: dict):
            const = 0.0
            free_vars = [i for i in range(self.n) if i not in fixed_vars]
            Q_work = Q.copy()

            # 常数补偿
            for i, vi in fixed_vars.items():
                const += Q_work[i, i] * vi

            # 固定-固定之间的二次项
            for i, vi in fixed_vars.items():
                for j, vj in fixed_vars.items():
                    if j > i:
                        const += Q_work[i, j] * vi * vj

            # 固定-自由的交互项转化为自由变量的偏置
            for idx in free_vars:
                bias = 0.0
                for j, vj in fixed_vars.items():
                    a, b = (min(idx, j), max(idx, j))
                    bias += Q_work[a, b] * vj
                Q_work[idx, idx] += bias

            Q_new = Q_work[np.ix_(free_vars, free_vars)]

            return Q_new, const, free_vars


        # 计算Roof Dual下界
        if fixed_vars is None:
            fixed_vars = {}

        Q_clamped, const, free_vars = clamp_qubo(self, Q, fixed_vars)
        m = len(free_vars)

        # 直接计算
        if m == 0:
            return const
        
        if m <= exact_limit:
            best = float("inf")
            for mask in range(1 << m):
                x = [(mask >> k) & 1 for k in range(m)]
                e = 0.0
                for i in range(m):
                    e += Q_clamped[i,i] * x[i]
                    for j in range(i+1, m):
                        e += Q_clamped[i,j] * x[i] * x[j]
                if e < best:
                    best = e
            return const + best

        #转换成 flow graph
        G = nx.DiGraph()

        source = "s"
        sink = "t"
        G.add_node(source)
        G.add_node(sink)

        # 添加自由变量节点
        for i in range(m):
            G.add_node(i)

        # 根据 Q_clamped 构造网络
        for i in range(m):
            G.add_edge(source, i, capacity=max(0, Q_clamped[i, i]))
            G.add_edge(i, sink, capacity=max(0, -Q_clamped[i, i]))
            for j in range(i + 1, m):
                w = Q_clamped[i, j] + Q_clamped[j, i]
                if not np.isclose(w, 0):
                    # 双向边表示割代价
                    G.add_edge(i, j, capacity=max(0, w))
                    G.add_edge(j, i, capacity=max(0, w))

        #计算 min-cut 对应的最大流
        flow_value, _ = nx.maximum_flow(G, source, sink)
        return const + flow_value
        
    def ub_local_search(self, Q: np.ndarray,fixed_vars=None, num_restarts=10) -> float:
        """
        使用局部搜索计算QUBO问题的上界
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            fixed_vars (dict): 已固定变量的字典，键为变量索引，值为固定值
            num_restarts (int): 随机重启次数，默认10
            
        Returns:
            best_energy(float): 计算得到的QUBO问题上界

        """
        # 如果没有固定变量，考虑所有变量
        if fixed_vars is None:
            fixed_vars = {}
        
        best_energy = np.inf
        
        # 定义QUBO能量函数
        def energy(z):
            total = 0
            for i in range(self.n):
                for j in range(i, self.n):
                    if i == j:
                        total += Q[i, i] * z[i]
                    else:
                        total += Q[i, j] * z[i] * z[j]
            return total
        
        # 多次随机重启
        for _ in range(num_restarts):
            # 初始化随机解
            z = np.random.randint(0, 2, self.n)
            
            # 应用固定变量
            for idx, val in fixed_vars.items():
                z[idx] = val
            
            current_energy = energy(z)
            improved = True
            
            # 局部搜索
            while improved:
                improved = False
                for i in range(self.n):
                    if i in fixed_vars:  # 跳过固定变量
                        continue
                    
                    # 尝试翻转当前变量
                    z[i] = 1 - z[i]
                    new_energy = energy(z)
                    
                    if new_energy < current_energy:
                        current_energy = new_energy
                        improved = True
                    else:
                        z[i] = 1 - z[i]
            
            # 更新最佳能量
            if current_energy < best_energy:
                best_energy = current_energy
        
        return best_energy
    
    def compute_w_bounds(self, Q: np.ndarray, k, l) -> Tuple[float, float]:
        """
        计算权重w的取值范围 [w_min, w_max]
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            k (int): 变量索引k
            l (int): 变量索引l
            
        Returns:
            w_min(float): 计算得到的权重w的最小值
            w_max(float): 计算得到的权重w的最大值

        """
        # 区分对角线元素和非对角线元素
        is_diagonal = (k == l)
        
        # 存储不同固定配置下的能量
        y_hat = {}
        y_bar = {}
        
        # 考虑所有可能的固定配置
        if is_diagonal:
            configs = [(0, 0), (1, 1)]
        else:
            configs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        # 计算每种固定配置下的能量边界
        for a, b in configs:
            fixed_vars = {k: a}
            if not is_diagonal:
                fixed_vars[l] = b
            
            y_hat[(a, b)] = self.lb_roof_dual(Q, fixed_vars)
            y_bar[(a, b)] = self.ub_local_search(Q, fixed_vars)
        
        # 计算w的边界
        if is_diagonal:
            delta_1 = y_hat[(0, 0)] - y_bar[(1, 1)]
            w_min = min(0, delta_1)

            delta_2 = y_bar[(0, 0)] - y_hat[(1, 1)]
            w_max = max(0, delta_2)
        else:
            min_other_1 = min(y_hat[(0, 0)], y_hat[(0, 1)], y_hat[(1, 0)])
            delta_1 = min_other_1 - y_bar[(1, 1)]
            w_min = min(0, delta_1)

            min_other_2 = min(y_bar[(0, 0)], y_bar[(0, 1)], y_bar[(1, 0)])
            delta_2 = min_other_2 - y_hat[(1, 1)]
            w_max = max(0, delta_2)
        
        return w_min, w_max

    def transition(self, Q: np.ndarray, action: Tuple[int, int]) -> np.ndarray:
        """
        状态转移函数:更新QUBO矩阵元素
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            action (Tuple[int, int]): 动作元组，包含要更新的元素索引
            
        Returns:
            new_Q(np.ndarray): 更新后的QUBO矩阵

        """
        i, j = action
        new_Q = Q.copy()

        # 使用roof_duality算法
        current_val = Q[i, j]
        w_min, w_max = self.compute_w_bounds(Q, i, j)
        if w_min <= -current_val <= w_max:
            w = -current_val
        elif current_val < 0:
            w = w_max
        elif current_val > 0:
            w = w_min
            
        new_Q[i, j] = current_val + w

        return new_Q

    def base_policy(self, Q: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        基础策略(贪心)
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            
        Returns:
            best_action(Tuple[int, int]): 计算得到的最优动作
            best_next(np.ndarray): 计算得到的最优下一个QUBO矩阵

        """
        best_DR = float('inf')
        best_action = None
        best_next = None
        
        actions = self.get_possible_actions(Q)
        for action in actions:
            next_Q = self.transition(Q, action)
            dr = self.dynamic_range(next_Q)
            if dr < best_DR:
                best_DR = dr
                best_action = action
                best_next = next_Q
        
        return best_action, best_next

    def rollout(self, Q: np.ndarray, steps: int) -> np.ndarray:
        """
        策略推演PR
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            steps (int): 推演步数
            
        Returns:
            final_Q(np.ndarray): 推演得到的最终QUBO矩阵

        """
        current = Q.copy()
        for _ in range(steps):
            _, current = self.base_policy(current)
        return current

    def compute_bound(self, Q: np.ndarray, remaining_steps: int) -> float:
        """
        计算状态r_bar的下界
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            remaining_steps (int): 剩余推演步数
            
        Returns:
            bound(float): 计算得到的状态r_bar的下界

        """
        future_Q = self.rollout(Q, remaining_steps)
        return self.dynamic_range(future_Q)

    def branch_and_bound(self) -> Tuple[np.ndarray, float]:
        """
        分支定界主算法
        Args:
            None
            
        Returns:
            best_Q(np.ndarray): 分支定界得到的最优QUBO矩阵
            best_DR(float): 分支定界得到的最优动态范围

        """
        queue = []
        counter = itertools.count()
        heapq.heappush(queue, (0, 0, next(counter), self.original_Q.copy(), []))
        
        while queue:
            neg_reward, step, _, current_Q, path = heapq.heappop(queue)
            current_DR = self.dynamic_range(current_Q)
            self.nodes_explored += 1
            
            if self.verbose:
                print(f"Step {step}, DR: {current_DR:.4f}, Path: {path}")
            
            # 到达终点或找到更好解
            if step == self.T:
                if current_DR < self.best_DR:
                    self.best_DR = current_DR
                    self.best_state = current_Q
                continue
            
            # 生成所有可能动作
            actions = self.get_possible_actions(current_Q)
            if self.verbose:
                print(f"  Possible actions: {len(actions)}")
            
            for action in actions:
                # 状态转移
                next_Q = self.transition(current_Q, action)
                next_DR = self.dynamic_range(next_Q)
                reward = current_DR - next_DR
                
                # 计算bound
                remaining_steps = self.T - step - 1
                bound = self.compute_bound(next_Q, min(remaining_steps, self.roll_depth))
                
                # 剪枝
                if bound >= self.best_DR:
                    self.nodes_pruned += 1
                    if self.verbose:
                        print(f"  Pruned action {action} (bound={bound:.2f} >= best={self.best_DR:.2f})")
                    continue
                
                # 新搜索节点
                new_path = path + [action]
                new_neg_reward = neg_reward - reward
                heapq.heappush(queue, (new_neg_reward, step+1, next(counter), next_Q, new_path))
        
        return self.best_state, self.best_DR

    def reduce_dynamic_range(self) -> Tuple[np.ndarray, float]:
        """
        动态范围缩减总执行程序
        Args:
            None
            
        Returns:
            best_Q(np.ndarray): 动态范围缩减后的最优QUBO矩阵
            best_DR(float): 动态范围缩减后的最优动态范围

        """
        # 初始化
        self.best_DR = self.dynamic_range(self.original_Q)
        self.best_state = self.original_Q.copy()
        
        # 执行分支定界程序
        best_Q, best_DR = self.branch_and_bound()
        
        if self.verbose:
            print(f"\nNodes explored: {self.nodes_explored}")
            print(f"Nodes pruned: {self.nodes_pruned}")
            print(f"Original DR: {self.dynamic_range(self.original_Q):.4f}")
            print(f"Reduced DR: {best_DR:.4f}")
        
        return best_Q, best_DR