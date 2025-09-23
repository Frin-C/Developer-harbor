import numpy as np
from scipy.stats import cauchy

def dataset(n: int = 8, seed = None) -> np.ndarray:
    '''
    生成数据集。

    Args:
        n: 数据点数量 (推荐 8 或 16)
        seed: 随机数种子

    Returns:
        X: 生成的数据集
    '''
    rng = np.random.default_rng(seed)
    
    # 生成n个独立同分布的二维点，服从各向同性的正态分布
    X = rng.normal(0, 0.1, (n, 2))
    
    # 创建两个簇
    X[:n//2, 0] = X[:n//2, 0] - 1
    X[n//2:, 0] = X[n//2:, 0] + 1
    
    # 添加两个异常值
    X[0] *= 100
    X[-1] *= 100

    return X

def binclus_qubo(n: int = 8, seed = None, kernel: str = 'linear') -> np.ndarray:
    '''
    生成 BINCLUS 的QUBO矩阵。

    Args:
        n: 数据点数量
        seed: 随机数种子
        kernel: 核函数类型 ('linear' 或 'rbf')

    Returns:
        Q: BINCLUS 的QUBO矩阵
    '''
    if kernel not in ['linear', 'rbf']:
        raise ValueError("kernel must be 'linear' or 'rbf'")
    
    # 生成数据集
    X = dataset(n, seed)

    if kernel == 'linear':
        # 计算线性核矩阵 K = X X^T
        K = X @ X.T
    else:
        # 计算RBF核矩阵
        gamma = 1.0 / X.shape[1]  # 默认gamma值
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist = np.linalg.norm(X[i] - X[j])
                K[i, j] = np.exp(-gamma * dist**2)
    
    # 中心化核矩阵
    ones = np.ones((n, n)) / n
    K_centered = K - ones @ K - K @ ones + ones @ K @ ones
    
    # 构建QUBO矩阵: min z^T (1^T K 1 - K) z
    Q = np.outer(np.ones(n), np.sum(K_centered, axis=1)) - K_centered
    
    # 转换为上三角形式（QUBO标准形式）
    Q_upper = np.triu(Q) + np.tril(Q, -1).T
    np.fill_diagonal(Q_upper, np.diag(Q))
    
    return Q_upper

def subsum_qubo(n: int = 8, seed = None) -> np.ndarray:
    '''
    生成 SUBSUM 的QUBO矩阵。

    Args:
        n: 数据点数量
        seed: 随机数种子
    
    Returns:
        Q: SUBSUM 的QUBO矩阵
    '''

    if seed is not None:
        np.random.seed(seed)
    
    # 生成整数集合，使用柯西分布
    Z = cauchy.rvs(size=n)
    A = np.floor(10 * np.abs(Z)).astype(int)
    
    # 确定子集大小k
    a = n / 5
    b = n / 2
    c = 4 * n / 5
    
    # 手动实现三角分布采样
    u = np.random.rand()
    if u < (b - a) / (c - a):
        k_sample = a + np.sqrt(u * (c - a) * (b - a))
    else:
        k_sample = c - np.sqrt((1 - u) * (c - a) * (c - b))
    
    k = int(np.round(k_sample))
    k = max(1, min(k, n-1))
    
    # 随机选择k个索引构成子集
    I = np.random.choice(n, size=k, replace=False)
    
    # 计算目标值T
    T = np.sum(A[I])

    # 构建二次项和线性项
    Q_quad = np.outer(A, A)
    linear_term = -2 * T * A
    
    # 组合成QUBO矩阵
    Q = Q_quad
    np.fill_diagonal(Q, np.diag(Q) + linear_term)
    
    # 转换为上三角形式
    Q_upper = np.triu(Q) + np.tril(Q, -1).T
    np.fill_diagonal(Q_upper, np.diag(Q))
    
    return Q_upper

def vecquant_qubo(n: int = 8, seed = None) -> np.ndarray:
    '''
    生成 VECQUANT 的QUBO矩阵。

    Args:
        n: 数据点数量
        seed: 随机数种子
    
    Returns:
        Q: VECQUANT 的QUBO矩阵
    '''
    # 生成数据集
    X = dataset(n, seed)

    # 计算距离矩阵
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(X[i] - X[j])
            D[i, j] = D[j, i] = 1 - np.exp(-dist**2/2)
    
    # 设置参数
    k, gamma = 4, 2
    alpha, beta = 1 / k, 1 / n

    # 构建二次项部分和线性项部分
    ones = np.ones((n, n))
    quadratic_part = gamma * ones - alpha * D
    linear_part = beta * np.sum(D, axis=1) - 2 * gamma * k * np.ones(n)
    
    # 组合成QUBO矩阵
    Q = quadratic_part
    np.fill_diagonal(Q, np.diag(Q) + linear_part)
    
    # 转换为上三角形式
    Q_upper = np.triu(Q) + np.tril(Q, -1).T
    np.fill_diagonal(Q_upper, np.diag(Q))
    
    return Q_upper
