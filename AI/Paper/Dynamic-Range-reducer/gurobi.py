import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_qubo(Q, time_limit=None, verbose=False):
    """
    使用Gurobi求解QUBO问题
    
    参数:
        Q (np.array): QUBO矩阵（n x n），应为上三角或对称矩阵
        time_limit (int): 求解时间限制（秒）
        verbose (bool): 是否显示求解器输出
        
    返回:
        dict: 包含状态、目标值和最优解向量的字典
    """
    try:
        # 验证Q矩阵
        n = Q.shape[0]
        if Q.shape != (n, n):
            raise ValueError("Q must be a square matrix")
            
        # 创建模型
        model = gp.Model("QUBO")
        
        # 设置输出级别
        model.setParam('OutputFlag', 1 if verbose else 0)
        if time_limit is not None:
            model.setParam('TimeLimit', time_limit)
        
        # 创建二值变量 (0或1)
        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        
        # 构建二次目标函数: sum_{i<=j} Q_ij * x_i * x_j
        # 对于i=j项: x_i*x_i = x_i (因为x_i是二值变量)
        # 对于i<j项: 只需计算上三角部分避免重复
        obj = gp.QuadExpr()
        
        # 对角线元素 (i=j)
        for i in range(n):
            obj += Q[i, i] * x[i]  # x[i]*x[i] = x[i] for binary vars
        
        # 非对角线元素 (i<j)
        for i in range(n):
            for j in range(i + 1, n):
                # 如果矩阵是对称的，使用Q[i,j]或Q[j,i]均可
                # 这里默认Q是上三角矩阵，所以使用Q[i,j]
                obj += Q[i, j] * x[i] * x[j]
        
        # 设置目标为最小化
        model.setObjective(obj, GRB.MINIMIZE)
        
        # 优化求解
        model.optimize()
        
        # 结果处理
        if model.status == GRB.OPTIMAL:
            solution = [round(x[i].X) for i in range(n)]  # 确保结果为整数
            return {
                "status": "OPTIMAL",
                "obj_value": model.objVal,
                "solution": solution
            }
        elif model.status == GRB.TIME_LIMIT:
            if model.SolCount > 0:  # 有时间限制但找到了可行解
                solution = [round(x[i].X) for i in range(n)]
                return {
                    "status": "TIME_LIMIT (feasible)",
                    "obj_value": model.objVal,
                    "solution": solution
                }
            return {"status": "TIME_LIMIT (no solution)"}
        elif model.status == GRB.INFEASIBLE:
            return {"status": "INFEASIBLE"}
        elif model.status == GRB.UNBOUNDED:
            return {"status": "UNBOUNDED"}
        else:
            return {"status": f"UNKNOWN_STATUS ({model.status})"}
            
    except gp.GurobiError as e:
        return {"status": f"GUROBI_ERROR: {str(e)}"}
    except Exception as e:
        return {"status": f"GENERAL_ERROR: {str(e)}"}
    