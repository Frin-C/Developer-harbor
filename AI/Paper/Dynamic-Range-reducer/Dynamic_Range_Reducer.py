import numpy as np
import math
import itertools
from typing import Tuple, List
from collections import deque, Counter
import bisect

class QUBODynamicRangeReducer:
    def __init__(self, Q: np.ndarray, T: int = 5, roll_depth: int = 2, 
                 policy_select: str = 'selection', branch_strategy: str = 'IMPACT', verbose: bool = False):
        """
        初始化QUBO动态范围缩减器
        
        Args:
            Q (np.ndarray): 原始QUBO矩阵(上三角形式)
            T (int): 最大搜索步数，默认5
            roll_depth (int): 策略推演深度，默认2
            policy_select (str): 策略选择('selection','mixed'或'base')，默认'selection'
            branch_strategy (str): 分支策略('ALL'或'IMPACT')，默认'IMPACT'
            verbose (bool): 是否打印详细过程，默认False

        """
        self.original_Q = Q.copy()
        self.n = Q.shape[0]
        self.T = T
        self.roll_depth = 0 if policy_select == 'base' else roll_depth if roll_depth < T else T
        self.policy_select = policy_select
        self.branch_strategy = branch_strategy
        self.verbose = verbose

        # current_Q 用于保存可被增量更新的当前矩阵副本
        self.current_Q = self.original_Q.copy()
        # 初始化增量统计结构以避免每次遍历完整矩阵计算 dynamic range
        self.value_counts = Counter()
        for i in range(self.n):
            for j in range(i, self.n):
                self.value_counts[float(Q[i, j])] += 1
        # 有序唯一值列表
        self.sorted_unique = sorted(self.value_counts.keys())
        # 缓存最小非零差值与最大差
        self._recompute_range_stats()
        self.original_DR = self.get_dynamic_range_from_stats()

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
        # 如果请求对当前维护的矩阵计算动态范围，使用增量统计以避免遍历
        if Q is self.current_Q:
            return self.get_dynamic_range_from_stats()

        # 否则回退到全矩阵计算（保持向后兼容）
        values = []
        for i in range(self.n):
            for j in range(i, self.n):
                values.append(Q[i, j])

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
    
    def _replace_value(self, old_val: float, new_val: float) -> None:
        """
        将值计数器中的旧值替换为新值，并更新有序唯一值列表
        
        Args:
            old_val (float): 要替换的旧值
            new_val (float): 要替换成的新值
        
        Returns:
            None

        """
        old_val = float(old_val)
        new_val = float(new_val)
        
        # 如果新旧值相同，无需操作
        if old_val == new_val:
            return
            
        # 处理旧值
        if self.value_counts[old_val] <= 1:
            # 删除键
            del self.value_counts[old_val]
            idx = bisect.bisect_left(self.sorted_unique, old_val)
            if idx < len(self.sorted_unique) and self.sorted_unique[idx] == old_val:
                self.sorted_unique.pop(idx)
        else:
            self.value_counts[old_val] -= 1
            
        # 处理新值
        self.value_counts[new_val] += 1
        if self.value_counts[new_val] == 1:
            bisect.insort(self.sorted_unique, new_val)
            
        self._recompute_range_stats()

    def _recompute_range_stats(self) -> None:
        """
        重新计算最大差与最小正差
        
        Returns:
            None

        """
        # 计算 max_diff 和 min_positive_diff
        if len(self.sorted_unique) < 2:
            self._max_diff = 0.0
            self._min_pos_diff = float('inf')
            return
        self._max_diff = self.sorted_unique[-1] - self.sorted_unique[0]
        # 找最小正差
        min_diff = float('inf')
        prev = self.sorted_unique[0]
        for v in self.sorted_unique[1:]:
            diff = v - prev
            if diff > 0 and diff < min_diff:
                min_diff = diff
            prev = v
        self._min_pos_diff = min_diff if min_diff < float('inf') else float('inf')

    def get_dynamic_range_from_stats(self) -> float:
        """
        从值计数器计算动态范围
        
        Returns:
            float: 计算得到的动态范围值

        """
        if self._min_pos_diff == float('inf') or self._min_pos_diff <= 0:
            return float('inf') if self._max_diff > 0 else 0.0
        return math.log2(self._max_diff / self._min_pos_diff)

    def _update_value_at(self, i: int, j: int, new_val: float) -> None:
        """
        在维护的 current_Q 上将 (i,j) 的值从旧值替换为 new_val，并更新统计结构。
        
        Args:
            i (int): 要更新的QUBO矩阵行索引
            j (int): 要更新的QUBO矩阵列索引
            new_val (float): 新值
        
        Returns:
            None

        """
        old = float(self.current_Q[i, j])
        new = float(new_val)
        if old == new:
            return
        # 上三角约定：只维护 i<=j 部分
        a, b = (i, j) if i <= j else (j, i)
        self._replace_value(old, new)
        self.current_Q[a, b] = new

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

    def clamp_qubo(self, Q: np.ndarray, fixed_vars: dict) -> Tuple[np.ndarray, float]:
        """
        固定变量并简化QUBO矩阵
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            fixed_vars (dict): 已固定变量的字典，键为变量索引，值为固定值

        Returns:
            Q_new(np.ndarray): 简化后的QUBO矩阵
            const(float): 固定变量引入的常数项
        """
        const = 0.0
        free_vars = [i for i in range(self.n) if i not in fixed_vars]
        Q_work = Q.copy()

        # 常数补偿项
        for i, vi in fixed_vars.items():
            for j, vj in fixed_vars.items():
                const += Q_work[i, j] * vi * vj

        # 固定-自由的交互项转化为自由变量的偏置
        for idx in free_vars:
            bias = 0.0
            for j, vj in fixed_vars.items():
                bias += Q_work[idx, j] * vj
                bias += Q_work[j, idx] * vj
            Q_work[idx, idx] += bias

        Q_new = Q_work[np.ix_(free_vars, free_vars)]
        return Q_new, const

    def lb_roof_duality(self, Q: np.ndarray, fixed_vars: dict = None) -> float:
        """
        使用Roof Dual下界计算QUBO矩阵.
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            fixed_vars (dict): 已固定变量的字典，键为变量索引，值为固定值
            
        Returns:
            lower_energy(float): 计算得到的Roof Dual下界
        """
        if fixed_vars is None:
            fixed_vars = {}

        Q_clamped, const_clamped = self.clamp_qubo(Q, fixed_vars)
        m = Q_clamped.shape[0]

        if m == 0:
            return float(const_clamped)

        try:
            from igraph import Graph
        except ImportError as e:
            raise ImportError(
                "igraph needs to be installed prior to running qubolite.lb_roof_dual(). You can "
                "install igraph with:\n'pip install igraph'"
            ) from e

        def to_posiform(Q_clamped: np.ndarray) -> tuple[np.ndarray, float]:
            posiform = np.zeros((2, m, m))
            # posiform[0] 包含 xi* xj 项，以及对角线上的 xi 项
            # posiform[1] 包含 xi*!xj 项，以及对角线上的 !xi 项
            lin = np.diag(Q_clamped)
            qua = np.triu(Q_clamped, 1)
            diag_ix = np.diag_indices_from(Q_clamped)
            qua_neg = np.minimum(qua, 0)
            posiform[0] = np.maximum(qua, 0)
            posiform[1] = -qua_neg
            posiform[0][diag_ix] = lin + qua_neg.sum(1)
            lin_ = posiform[0][diag_ix].copy()  # =: c'
            lin_neg = np.minimum(lin_, 0)
            posiform[1][diag_ix] = -lin_neg
            posiform[0][diag_ix] = np.maximum(lin_, 0)
            const = lin_neg.sum()
            return posiform, const

        def to_flow_graph(P):
            n = P.shape[1]
            G = Graph(directed=True)
            vertices = np.arange(n + 1)
            negated_vertices = np.arange(n + 1, 2 * n + 2)
            # 流图的所有顶点
            all_vertices = np.concatenate([vertices, negated_vertices])
            G.add_vertices(all_vertices)
            # 包含节点 x0 的顶点数组
            n0 = np.kron(vertices[1:][:, np.newaxis], np.ones(n, dtype=int))
            np.fill_diagonal(n0, np.zeros(n))
            nn0 = np.kron(negated_vertices[1:][:, np.newaxis], np.ones(n, dtype=int))
            np.fill_diagonal(nn0, (n + 1) * np.ones(n))
            # 不包含节点 x0 的顶点数组
            n1 = np.kron(np.ones(n, dtype=int)[:, np.newaxis], vertices[1:])
            nn1 = np.kron(np.ones(n, dtype=int)[:, np.newaxis], negated_vertices[1:])

            n0_nn1 = np.stack((n0, nn1), axis=-1) # 从 ni 到 !nj 的边
            n1_nn0 = np.stack((n1, nn0), axis=-1) # 从 nj 到 !ni 的边
            n0_n1 = np.stack((n0, n1), axis=-1) # 从 ni 到 nj 的边
            nn1_nn0 = np.stack((nn1, nn0), axis=-1) # 从 !nj 到 !ni 的边
            pos_indices = np.invert(np.isclose(P[0], 0))
            neg_indices = np.invert(np.isclose(P[1], 0))
            # 将容量设置为正形参数的一半
            capacities = 0.5 * np.concatenate([P[0][pos_indices], P[0][pos_indices],
                                            P[1][neg_indices], P[1][neg_indices]])
            edges = np.concatenate([n0_nn1[pos_indices], n1_nn0[pos_indices],
                                    n0_n1[neg_indices], nn1_nn0[neg_indices]])
            G.add_edges(edges)
            return G, capacities

        P, const = to_posiform(Q_clamped)
        G, capacities = to_flow_graph(P)
        mf = G.maxflow(0, m + 1, capacity=list(capacities))
        v = mf.value
        return const + v + const_clamped

    def lb_negative(self, Q: np.ndarray, fixed_vars: dict = None) -> float:
        """
        使用负元素下界计算QUBO问题的下界
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            fixed_vars (dict): 已固定变量的字典，键为变量索引，值为固定值
            
        Returns:
            lower_energy(float): 计算得到的负权重下界
        """
        if fixed_vars is None:
            fixed_vars = {}

        Q_clamped, const = self.clamp_qubo(Q, fixed_vars)
        m = Q_clamped.shape[0]
        n = Q_clamped.shape[1]

        if m == 0:
            return const
        
        lower_bound = 0.0
        
        # 计算二次项贡献
        for i in range(m):
            for j in range(i, n):
                w_ij = Q_clamped[i, j]
                if w_ij < 0:
                    lower_bound += w_ij
        return const + lower_bound

    def ub_local_search(self, Q: np.ndarray, fixed_vars=None, num_restarts: int = 10) -> float:
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

            y_hat[(a, b)] = self.ub_local_search(Q, fixed_vars)
            y_bar[(a, b)] = self.lb_roof_duality(Q, fixed_vars)

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

        # 使用roof_duality or negative算法
        current_val = Q[i, j]
        w_min, w_max = self.compute_w_bounds(Q, i, j)

        if w_min <= -current_val <= w_max:
            w = -current_val
            new_Q[i, j] = current_val + w
            # 如果我们维护了 current_Q，并且 Q 就是 current_Q，则增量更新统计
            if Q is self.current_Q:
                self._update_value_at(i, j, new_Q[i, j])
            else:
                # 否则将 current_Q 同步为 new_Q（批量更新）
                self._bulk_sync_current_Q(new_Q)
            return new_Q

        if current_val < 0:
            w = w_max
        else:
            w = w_min
            
        new_Q[i, j] = current_val + w
        if Q is self.current_Q:
            self._update_value_at(i, j, new_Q[i, j])
        else:
            # 否则将 current_Q 同步为 new_Q（批量更新）
            self._bulk_sync_current_Q(new_Q)
        return new_Q

    def _bulk_sync_current_Q(self, new_Q: np.ndarray) -> None:
        """
        当全矩阵被替换时，将 current_Q 同步为 new_Q，并用增量更新统计（逐元素对比）
        
        Args:
            new_Q (np.ndarray): 新的QUBO矩阵
        
        Returns:
            None

        """
        for i in range(self.n):
            for j in range(i, self.n):
                old = float(self.current_Q[i, j])
                new = float(new_Q[i, j])
                if old != new:
                    self._replace_value(old, new)
                    self.current_Q[i, j] = new

    def policy(self, Q: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        贪心算法(Greedy Algorithm)单步短视的优化DR
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            
        Returns:
            best_action(Tuple[int, int]): 计算得到的最优动作
            best_next(np.ndarray): 计算得到的最优下一个QUBO矩阵

        """
        best_DR = float('inf')
        best_action = None
        best_next = None
        if self.n > 5:
            # get possible actions based on IMPACT strategy
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

            actions = list(actions)
        else:
            actions = self.get_possible_actions(Q)

        for action in actions:
            next_Q = self.transition(Q, action)
            dr = self.dynamic_range(next_Q)
            if dr < best_DR:
                best_DR = dr
                best_action = action
                best_next = next_Q
        return best_action, best_next

    def rollout(self, Q: np.ndarray, remaining_steps: int) -> float:
        """
        通过策略推演PR，计算状态r_bar的下界
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            remaining_steps (int): 剩余推演步数
            
        Returns:
            bound(float): 计算得到的状态r_bar的下界

        """

        future_Q = Q.copy()
        for _ in range(remaining_steps):
            _, future_Q = self.policy(future_Q)
        return self.dynamic_range(future_Q)

    def branch_and_bound(self)-> Tuple[np.ndarray, float]:
        """
        分支定界主算法
        Args:
            None
            
        Returns:
            best_Q(np.ndarray): 分支定界得到的最优QUBO矩阵
            best_DR(float): 分支定界得到的最优动态范围

        """
        queue = deque()
        counter = itertools.count()
        queue.append((0, 0, next(counter), self.original_Q.copy(), []))
        
        while queue:
            neg_reward, step, _, current_Q, path = queue.popleft()
            current_DR = self.dynamic_range(current_Q)
            self.nodes_explored += 1
            
            if self.verbose:
                print(f"Step {step}, DR: {current_DR:.4f}, Path: {path}")
            
            # 更新最优解
            if current_DR < self.best_DR:
                self.best_DR = current_DR
                self.best_state = current_Q


            if step == self.T:
                continue



            # 检索超过roll_depth时，切换为base policy
            if step == self.roll_depth and (self.policy_select == 'mixed' or self.policy_select == 'base'):
                self.policy_select = 'base'
                continue
            
            # 生成所有可能动作
            actions = self.get_possible_actions(current_Q)
            # 检索所有可能动作
            for action in actions:
                next_Q = self.transition(current_Q, action)
                next_DR = self.dynamic_range(next_Q)
                reward = current_DR - next_DR
                
                # 计算bound
                remaining_steps = self.T - step - 1
                remaining_step = self.roll_depth - step - 1

                bound = self.rollout(
                    next_Q,
                    min(remaining_steps, self.roll_depth) if self.policy_select == 'selection' else max(remaining_step, 1)
                )
                
                # 剪枝
                if bound >= self.best_DR:
                    self.nodes_pruned += 1
                    if self.verbose:
                        print(f"Pruned action {action} (bound={bound:.2f} >= best={self.best_DR:.2f})")
                    continue
                
                # 新搜索节点
                new_path = path + [action]
                new_neg_reward = neg_reward - reward
                queue.append((new_neg_reward, step+1, next(counter), next_Q, new_path))
                
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
        self.best_DR = self.original_DR
        self.best_state = self.original_Q.copy()
        
        # 执行分支定界程序
        best_Q, best_DR = self.branch_and_bound()
        # 后续贪心策略计算
        if self.policy_select == 'base':
            for _ in range(self.roll_depth+1, self.T+1):
                best_action, best_Q = self.policy(best_Q)
                best_DR = self.dynamic_range(best_Q)
                self.nodes_explored += 1
                if self.verbose:
                    print(f"Step {_}, DR: {best_DR:.4f}, Action: {best_action}")
        
        if self.verbose:
            print(f"\nNodes explored: {self.nodes_explored}")
            print(f"Nodes pruned: {self.nodes_pruned}")
            print(f"Original DR: {self.original_DR:.4f}")
            print(f"Reduced DR: {best_DR:.4f}")
        return best_Q, best_DR