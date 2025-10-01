import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from Dynamic_Range_Reducer import QUBODynamicRangeReducer
import data_producer as dp
from gurobi import solve_qubo

print("\033c", end="")
# 生成数据
"""
qubo_n4 = []
qubo_n8 = []
for k in range(20):
    qubo_n4.append(dp.subsum_qubo(4, k))
    qubo_n8.append(dp.subsum_qubo(8, k))

np.save('qubo_n4.npy', qubo_n4)
np.save('qubo_n8.npy', qubo_n8)
"""
# 读取数据并计算
qubo_n4 = np.load('qubo_n4.npy', allow_pickle=True)
qubo_n8 = np.load('qubo_n8.npy', allow_pickle=True)

# 记录修剪比例
pruned_state_n4 = []
pruned_state_n8 = []
for i in range(2, 6):
    print(f"Param {i}/5")
    for k in range(20):
        print(f"  Iteration {k+1}/20")
        for j in range(2, 7):
            print(f"    Horizon {j}/6")
            reducer4 = QUBODynamicRangeReducer(qubo_n4[k], j, i, 'mixed', verbose=False)
            reduced_Q, final_DR = reducer4.reduce_dynamic_range()
            result = solve_qubo(qubo_n4[k], time_limit=30, verbose=False)
            result_reduced = solve_qubo(reduced_Q, time_limit=30, verbose=False)
            # 检查解是否一致
            if result['solution'] != result_reduced['solution']:
                if np.array(result['solution']).T@reduced_Q@np.array(result['solution']) != np.array(result_reduced['solution']).T@reduced_Q@np.array(result_reduced['solution']):
                    print("Error: Solutions do not match!")
                    print(f"Original: {result['solution']}, Reduced: {result_reduced['solution']}")
                    print(f"Original Objective: {np.array(result['solution']).T@reduced_Q@np.array(result['solution'])}, Reduced Objective: {np.array(result_reduced['solution']).T@reduced_Q@np.array(result_reduced['solution'])}")
            print(f"      Pruned {reducer4.nodes_pruned} out of {reducer4.nodes_explored} nodes.")

            pruned_state_n4.append(reducer4.nodes_pruned / (reducer4.nodes_explored + reducer4.nodes_pruned))

            reducer8 = QUBODynamicRangeReducer(qubo_n8[k], j, i, 'mixed')
            reduced_Q, final_DR = reducer8.reduce_dynamic_range()
            result = solve_qubo(qubo_n8[k], time_limit=30, verbose=False)
            result_reduced = solve_qubo(reduced_Q, time_limit=30, verbose=False)
            # 检查解是否一致
            if result['solution'] != result_reduced['solution']:
                if np.array(result['solution']).T@reduced_Q@np.array(result['solution']) != np.array(result_reduced['solution']).T@reduced_Q@np.array(result_reduced['solution']):
                    print("Error: Solutions do not match!")
                    print(f"Original: {result['solution']}, Reduced: {result_reduced['solution']}")
                    print(f"Original Objective: {np.array(result['solution']).T@reduced_Q@np.array(result['solution'])}, Reduced Objective: {np.array(result_reduced['solution']).T@reduced_Q@np.array(result_reduced['solution'])}")
            print(f"      Pruned {reducer8.nodes_pruned} out of {reducer8.nodes_explored} nodes.")
            pruned_state_n8.append(reducer8.nodes_pruned / (reducer8.nodes_explored + reducer8.nodes_pruned))
pruned_state_n4 = np.array(pruned_state_n4)
pruned_state_n8 = np.array(pruned_state_n8)

horizons = np.tile(np.arange(2, 7), 80)
params = np.repeat([2, 3, 4, 5], 100)
# 构造成 DataFrame
df_n8 = pd.DataFrame({
    "Horizon": horizons,
    "Fraction of pruned states": pruned_state_n4,
    "Param": params
})

df_n16 = pd.DataFrame({
    "Horizon": horizons,
    "Fraction of pruned states": pruned_state_n8,
    "Param": params
})

# 颜色配置
colors = ["#d7bcad", "#c58ca3", "#8c658c", "#3f2b56"]

# Pruned states fraction plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# 左边子图
sns.boxplot(
    data=df_n8[df_n8["Param"].isin([2, 3, 4, 5])],
    x="Horizon",
    y="Fraction of pruned states",
    hue="Param",
    palette=colors,
    linewidth=1,
    fliersize=3,
    ax=axes[0]
)
# 右边子图
sns.boxplot(
    data=df_n16[df_n16["Param"].isin([2, 3, 4, 5])],
    x="Horizon",
    y="Fraction of pruned states",
    hue="Param",
    palette=colors,
    linewidth=1,
    fliersize=3,
    ax=axes[1]
)
# 美化
for ax in axes:
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Fraction of pruned states")
    ax.legend(title="", loc="upper left")
plt.tight_layout()
plt.savefig('Fraction_of_pruned_states.png', dpi=300, bbox_inches='tight')

# DR reduction plot
strategies = ["base", 2, 4, "selection"]
# 记录 DR_reduction
DR_all_n4 = []
DR_all_n8 = []
DR_impact_n4 = []
DR_impact_n8 = []
for i in ["base", 2, 4, "selection"]:
    print(f"Strategy {i}")
    for k in range(20):
        print(f"  Iteration {k+1}/20")
        for j in [0, 2, 3, 4, 5, 6]:
            print(f"    Horizon {j}/6")
            # 记录 DR_reduction
            if i == "base":
                '''
                reducer4_all = QUBODynamicRangeReducer(qubo_n4[k], j, 0, 'base', 'ALL', verbose=False)
                reduced_Q, final_DR = reducer4_all.reduce_dynamic_range()
                initial_DR = reducer4_all.original_DR
                DR_all_n4.append((initial_DR - final_DR) / initial_DR)

                reducer8_all = QUBODynamicRangeReducer(qubo_n8[k], j, 0, 'base', 'ALL', verbose=False)
                reduced_Q, final_DR = reducer8_all.reduce_dynamic_range()
                initial_DR = reducer8_all.original_DR
                DR_all_n8.append((initial_DR - final_DR) / initial_DR)
                '''
                reducer4_impact = QUBODynamicRangeReducer(qubo_n4[k], j, 0, 'base', 'IMPACT', verbose=False)
                reduced_Q, final_DR = reducer4_impact.reduce_dynamic_range()
                initial_DR = reducer4_impact.original_DR
                DR_impact_n4.append((initial_DR - final_DR) / initial_DR)

                reducer8_impact = QUBODynamicRangeReducer(qubo_n8[k], j, 0, 'base', 'IMPACT', verbose=False)
                reduced_Q, final_DR = reducer8_impact.reduce_dynamic_range()
                initial_DR = reducer8_impact.original_DR
                DR_impact_n8.append((initial_DR - final_DR) / initial_DR)
            elif i == "selection":
                '''
                reducer4_all = QUBODynamicRangeReducer(qubo_n4[k], j, 2, 'selection', 'ALL', verbose=False)
                reduced_Q, final_DR = reducer4_all.reduce_dynamic_range()
                initial_DR = reducer4_all.original_DR
                DR_all_n4.append((initial_DR - final_DR) / initial_DR)

                reducer8_all = QUBODynamicRangeReducer(qubo_n8[k], j, 2, 'selection', 'ALL', verbose=False)
                reduced_Q, final_DR = reducer8_all.reduce_dynamic_range()
                initial_DR = reducer8_all.original_DR
                DR_all_n8.append((initial_DR - final_DR) / initial_DR)
                '''
                reducer4_impact = QUBODynamicRangeReducer(qubo_n4[k], j, 2, 'selection', 'IMPACT', verbose=False)
                reduced_Q, final_DR = reducer4_impact.reduce_dynamic_range()
                initial_DR = reducer4_impact.original_DR
                DR_impact_n4.append((initial_DR - final_DR) / initial_DR)

                reducer8_impact = QUBODynamicRangeReducer(qubo_n8[k], j, 2, 'selection', 'IMPACT', verbose=False)
                reduced_Q, final_DR = reducer8_impact.reduce_dynamic_range()
                initial_DR = reducer8_impact.original_DR
                DR_impact_n8.append((initial_DR - final_DR) / initial_DR)

            else:
                '''
                reducer4_all = QUBODynamicRangeReducer(qubo_n4[k], j, i, 'mixed', 'ALL', verbose=False)
                reduced_Q, final_DR = reducer4_all.reduce_dynamic_range()
                initial_DR = reducer4_all.original_DR
                DR_all_n4.append((initial_DR - final_DR) / initial_DR)

                reducer8_all = QUBODynamicRangeReducer(qubo_n8[k], j, i, 'mixed', 'ALL', verbose=False)
                reduced_Q, final_DR = reducer8_all.reduce_dynamic_range()
                initial_DR = reducer8_all.original_DR
                DR_all_n8.append((initial_DR - final_DR) / initial_DR)
                '''
                reducer4_impact = QUBODynamicRangeReducer(qubo_n4[k], j, i, 'mixed', 'IMPACT', verbose=False)
                reduced_Q, final_DR = reducer4_impact.reduce_dynamic_range()
                initial_DR = reducer4_impact.original_DR
                DR_impact_n4.append((initial_DR - final_DR) / initial_DR)

                reducer8_impact = QUBODynamicRangeReducer(qubo_n8[k], j, i, 'mixed', 'IMPACT', verbose=False)
                reduced_Q, final_DR = reducer8_impact.reduce_dynamic_range()
                initial_DR = reducer8_impact.original_DR
                DR_impact_n8.append((initial_DR - final_DR) / initial_DR)

horizons = np.tile([0, 2, 3, 4, 5, 6], 80)
strategies = np.repeat(["$\\dot{\\pi}$ (baseline)",
                        "$\\bar{\\pi}_2$", 
                        "$\\bar{\\pi}_4$", 
                        "$\\tilde{\\pi}$"], 120)

# 设置颜色
colors = ["#a8c686", "#63b0a0", "#527ba4", "#2f2e63","#d7bcad", "#c58ca3", "#8c658c", "#3f2b56"]
# 构造成 DataFrame
'''
df_all_n4 = pd.DataFrame({
    "Horizon": horizons,
    "Relative DR reduction": DR_all_n4,
    "Strategy": strategies
})
df_all_n8 = pd.DataFrame({
    "Horizon": horizons,
    "Relative DR reduction": DR_all_n8,
    "Strategy": strategies
})
'''
df_impact_n4 = pd.DataFrame({
    "Horizon": horizons,
    "Relative DR reduction": DR_impact_n4,
    "Strategy": strategies
})
df_impact_n8 = pd.DataFrame({
    "Horizon": horizons,
    "Relative DR reduction": DR_impact_n8,
    "Strategy": strategies
})
# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
# 子图数据划分
subsets = [
    # df_all_n4[df_all_n4["Strategy"].isin(["$\\dot{\\pi}$ (baseline)", "$\\bar{\\pi}_2$", "$\\bar{\\pi}_4$", "$\\tilde{\\pi}$"])],
    # df_all_n8[df_all_n8["Strategy"].isin(["$\\dot{\\pi}$ (baseline)", "$\\bar{\\pi}_2$", "$\\bar{\\pi}_4$", "$\\tilde{\\pi}$"])],
    df_impact_n4[df_impact_n4["Strategy"].isin(["$\\dot{\\pi}$ (baseline)", "$\\bar{\\pi}_2$", "$\\bar{\\pi}_4$", "$\\tilde{\\pi}$"])],
    df_impact_n8[df_impact_n8["Strategy"].isin(["$\\dot{\\pi}$ (baseline)", "$\\bar{\\pi}_2$", "$\\bar{\\pi}_4$", "$\\tilde{\\pi}$"])]
]
for ax, subset in zip(axes.flatten(), subsets):
    # 根据子图索引选择不同的配色
    color_idx = 0 if ax == axes[0] else 4
    sns.boxplot(
        data=subset,
        x="Horizon",
        y="Relative DR reduction",
        hue="Strategy",
        palette=colors[color_idx:color_idx+4],
        linewidth=1,
        fliersize=3,
        ax=ax
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Relative DR reduction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="", loc="upper left")
plt.tight_layout()
plt.savefig('Relative_DR_reduction.png', dpi=300, bbox_inches='tight')
plt.show()

