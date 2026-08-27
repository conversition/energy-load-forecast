# 电力负荷预测(第四届世界科学智能大赛 · 全国电力负荷预测竞赛)

基于边界条件与气象数据,预测 **D+1 全天 96 个 15 分钟粒度实时电价**,并在容量、功率、充放电频次约束下生成充放电计划,实现**日均收益最大化**。

## 🎯 项目目标

- **电价预测**:多源时序数据 → D+1 96 点实时电价回归预测
- **策略优化**:在储能容量/功率/充放电频次约束下,生成每日充放电计划,以真实业务收益(而非 RMSE)为最终优化指标

## 🛠 技术栈

Python · LightGBM · GradientBoostingRegressor · Pandas · NumPy · Xarray · scikit-learn

## 🧠 技术方案

### 1. 多源数据清洗与特征工程
- 十万级多源时序数据清洗:对齐边界条件与电价标签时序
- 气象特征空间平均、交互特征构造;业务/物理特征缺失剔除、时间戳去重排序、插值
- 排列重要性 + 相关性矩阵特征剪枝,缓解模型过拟合

### 2. 模型演进(50+ 版本迭代)
- **基线**:GBDT(`sklearn_baseline`)、LSTM(`lstm_baseline`)
- **主线**:GradientBoostingRegressor 基础特征回归 → 引入气象特征后 LightGBM 捕捉气象突变极端电价
- **融合**:GBDT + LightGBM 融合模型,以**真实业务收益**而非 RMSE 作为指标选择最优权重
- **进阶探索**:分位数回归、非对称损失、多滞后特征、时序切分、天气深度特征、LSTM-LGB 融合等(见 `v00~v51` 迭代脚本)

### 3. 策略与评估
- 充放电策略约束建模(容量/功率/频次),`v*_profit_eval` 系列脚本以收益评估模型

## 📁 代码结构

```
├── sklearn_baseline*.py      # GBDT 基线 + 迭代
├── lstm_baseline.py          # LSTM 基线
├── feature_analysis.py       # 特征分析
├── v0_baseline.py ~ v51_*    # 50+ 版本迭代(v1 动态窗口 → v51 非对称损失)
├── output/                   # 各版本输出(不入库)
└── README.md
```

## 🚀 运行

```bash
# 基线
python sklearn_baseline.py
python lstm_baseline.py

# 各版本实验(示例)
python v28_final.py
python v36_final_robust.py
python v46_quantile_final.py
```

> 竞赛数据集来自 ModelScope:Datawhale/AI_camp_energy_2026(数据文件不入库)。

## 📄 License

Apache License 2.0
