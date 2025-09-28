import numpy as np
import math
import itertools
import networkx as nx
from typing import Tuple, List
from collections import deque

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

    def clamp_qubo(self, Q: np.ndarray, fixed_vars: dict) -> Tuple[np.ndarray, float]:
        """
        固定变量并简化QUBO矩阵
        Args:
            Q (np.ndarray): 输入QUBO矩阵
            fixed_vars (dict): 已固定变量的字典，键为变量索引，值为固定值

        Returns:
            Q_new(np.ndarray): 简化后的QUBO矩阵
            const(float): 固定变量引入的常数项
            free_vars(List[int]): 未固定变量的索引列表
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
        if fixed_vars is None:
            fixed_vars = {}

        Q_clamped, const = self.clamp_qubo(Q, fixed_vars)
        m = Q_clamped.shape[0]

        if m == 0:
            return float(const)

        if m <= exact_limit:
            best = None
            for mask in range(1 << m):
                x = np.array([(mask >> i) & 1 for i in range(m)], dtype=float)
                val = x @ Q_clamped @ x
                if best is None or val < best:
                    best = val
            return float(const + best)

        # 对称化 Q（确保 Q 为对称矩阵）
        Qs = (Q_clamped + Q_clamped.T) / 2.0

        # 翻转工具：对给定 flip_mask (bool array)，计算变换后的 Q 和常数增量
        def apply_flips(Qmat: np.ndarray, flip_mask: np.ndarray):
            # Qmat assumed symmetric
            mloc = Qmat.shape[0]
            M = np.diag(np.where(flip_mask, -1.0, 1.0))   # M_ii = 1 or -1
            c = flip_mask.astype(float)                   # c_i = 1 if flipped else 0
            # Q' = M Q M
            Qp = M @ Qmat @ M
            # 线性项来自 2 * M^T Q c  （M 对角可简化）
            L = 2.0 * (M @ (Qmat @ c))
            # 将线性项并入对角
            Qp = Qp.copy()
            for i in range(mloc):
                Qp[i, i] += L[i]
            # 常数项 c^T Q c
            const_add = float(c @ (Qmat @ c))
            return Qp, const_add

        # 目标：最小化正的 off-diagonal 权重之和（越小越好）
        def positive_offdiag_sum(Qmat: np.ndarray):
            s = 0.0
            for i in range(Qmat.shape[0]):
                for j in range(i + 1, Qmat.shape[0]):
                    w = Qmat[i, j]
                    if w > 0:
                        s += w
            return s

        # 贪心翻转：每步尝试翻转单个变量，若能减少正权重总和则保留，直到无改进
        flip_mask = np.zeros(m, dtype=bool)
        Qcurrent = Qs.copy()
        const_extra = 0.0
        improved = True
        while improved:
            improved = False
            best_reduction = 0.0
            best_i = -1
            best_Q = None
            best_const_add = 0.0
            base_score = positive_offdiag_sum(Qcurrent)
            for i in range(m):
                fm = flip_mask.copy()
                fm[i] = ~fm[i]
                Qp, cadd = apply_flips(Qs, fm)  # compute full transformed Qwrt original symmetrical Qs
                score = positive_offdiag_sum(Qp)
                reduction = base_score - score
                if reduction > best_reduction + 1e-12:
                    best_reduction = reduction
                    best_i = i
                    best_Q = Qp
                    best_const_add = cadd
            if best_i >= 0:
                # 接受该翻转
                flip_mask[best_i] = ~flip_mask[best_i]
                Qcurrent = best_Q
                const_extra = float(best_const_add)  # note: apply_flips computes c^T Q c for full flip_mask; we overwrite
                improved = True

        # apply_flips returned Qp that already absorbs linear terms into diagonals, and const_extra is c^T Q c
        # 需要注意：const_extra 是基于 Qs 的常数增量，Qs 对应的是 Q_clamped（已包含原 const），
        # 所以整体常数应累加
        const_total = const + const_extra

        # 现在 Qcurrent 为翻转后矩阵（已并入线性项到对角）
        # 再构造流网络：把负的二次项（w_ij < 0）转为边容量，正项尽量已被翻转为负
        G = nx.DiGraph()
        source = 's'
        sink = 't'
        G.add_node(source)
        G.add_node(sink)
        for i in range(m):
            G.add_node(i)

        # 对角项处理：正对角作为 source->i，负对角作为 i->sink
        for i in range(m):
            a_i = float(Qcurrent[i, i])
            if a_i > 0:
                G.add_edge(source, i, capacity=a_i)
            elif a_i < 0:
                G.add_edge(i, sink, capacity=-a_i)

        # 二次项处理：负交互（w < 0）转为节点间容量（双向）
        for i in range(m):
            for j in range(i + 1, m):
                w = float(Qcurrent[i, j])
                if w < 0:
                    cap = -w
                    if G.has_edge(i, j):
                        G[i][j]['capacity'] += cap
                    else:
                        G.add_edge(i, j, capacity=cap)
                    if G.has_edge(j, i):
                        G[j][i]['capacity'] += cap
                    else:
                        G.add_edge(j, i, capacity=cap)
                else:
                    # 如果仍为正（未被翻转消除），可以做简单的上界分解以保守地把一部分移到对角
                    # 这里把正项 w 分解为：增加对角 w/2 到每个节点（等价于添加 linear bias），
                    # 并在常数上不改变（这是近似，不改变可行性，但会得到较弱的补充）
                    # 这样可以降低图中未处理的正权重对下界的影响
                    half = w / 2.0
                    # 把一半加入对角（相当于把 w x_i x_j ~ half*x_i + half*x_j - half*|x_i-x_j| 的一部分）
                    # 这是近似处理，目的是避免忽略该正权重所带来的松弛过大
                    if G.has_edge(source, i):
                        G[source][i]['capacity'] += half
                    else:
                        G.add_edge(source, i, capacity=half)
                    if G.has_edge(source, j):
                        G[source][j]['capacity'] += half
                    else:
                        G.add_edge(source, j, capacity=half)
                    # 同时把一半也作为从节点到汇的可能量（对称处理，以稳健性为主）
                    if G.has_edge(i, sink):
                        G[i][sink]['capacity'] += half
                    else:
                        G.add_edge(i, sink, capacity=half)
                    if G.has_edge(j, sink):
                        G[j][sink]['capacity'] += half
                    else:
                        G.add_edge(j, sink, capacity=half)
                    # 该近似并不会完美恢复原二次项，但通常能获得比完全忽略更紧的下界

        cut_value, (S, T) = nx.minimum_cut(G, source, sink, capacity='capacity')
        
        return float(const + cut_value)

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
            y_bar[(a, b)] = self.lb_negative(Q, fixed_vars)

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
            return new_Q

        if current_val < 0:
            w = w_max
        else:
            w = w_min
            
        new_Q[i, j] = current_val + w
        return new_Q

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
                    min(remaining_steps, self.T) if self.policy_select == 'selection' else max(remaining_step, 1)
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
        self.best_DR = self.dynamic_range(self.original_Q)
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
            print(f"Original DR: {self.dynamic_range(self.original_Q):.4f}")
            print(f"Reduced DR: {best_DR:.4f}")
        return best_Q, best_DR