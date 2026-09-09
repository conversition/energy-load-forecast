# RESULTS:收益导向的模型演进记录

> 任务:D+1 全天 96 个 15min 粒度实时电价预测 + 约束下充放电计划,**目标函数是日均收益,不是 RMSE**。

## 核心洞察

> **RMSE 最优 ≠ 收益最优。**
> 电价预测的价值不在"预测得准",而在"预测误差的**方向**是否导致错误决策"——
> 低估电价会错失放电良机,高估电价会导致错误充电亏损。因此:

- 评估体系从"误差指标"切换到 **`simulate_true_validation_profit`(真实收益回测)**
- 引入 **分位数回归(P10 / P50 / P90)**,用价格区间驱动策略,而非点预测
- 引入 **非对称损失**:对低估/高估赋予不同权重,直接对齐业务损失方向(代码见 `v51_asymmetric_loss.py`)

## 可验证的技术演进线(脚本即证据)

| 阶段 | 脚本 | 技术点 |
|---|---|---|
| 基线 | `sklearn_baseline.py` / `lstm_baseline.py` | GBDT / LSTM 双基线 |
| 稳定版 | `v4_stable_version.py` / `v6_with_interaction.py` | 特征交互、动态时段 |
| 特征工程 | `v10_correlation_analysis.py` / `v11_multicollinearity_optimized.py` | 相关性矩阵(实测 r=0.801 共线性对)、剪枝 |
| 天气增强 | `v26_weather.py` → `v34_robust_weather.py` | 气象特征引入与鲁棒化 |
| 收益导向评估 | `v31_profit_eval_fix.py` / `v32_true_profit.py` / `v33_cv_selection.py` | 以真实收益做交叉验证选模型 |
| 量化回归 | `v38_quantile_regression.py` → `v46_quantile_final.py` | P10/P50/P90 分位数框架 |
| 非对称损失 | `v51_asymmetric_loss.py` | 低估/高估差异化权重 |
| 深度学习融合 | `v49_lstm_temporal.py` / `v50_lstm_lgb_fusion.py` | LSTM 时序 + LightGBM 融合 |

## 输出格式

`output/output.csv`:`times / P10 / P90 / median / power`(P10/P90 为置信区间,power 为充放电计划)。

## 结果数据【待补】

> 诚实边界:当前仓库未附带最终收益数字(需在竞赛数据上复跑)。
> 复跑方式:

```bash
# 训练 + 回测收益(以 v46/v51 为例)
python v46_quantile_final.py
python v51_asymmetric_loss.py
# 输出:output/output.csv + 日志中的 test_profit
```

复跑后将收益对比表补入本节(基线 vs 量化回归 vs 非对称损失)。

## 竞赛数据说明

竞赛数据来自 ModelScope(`Datawhale/AI_camp_energy_2026`),体积大未入库;`output/` 为运行产物亦不入库,详见仓库 `.gitignore`。
