"""
v12 交叉验证评估 + 残差分析
1. 5折交叉验证获得更稳定的RMSE评估
2. 分析预测残差分布，找出未被捕捉的模式
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')

# v11 优化后的7个特征
feature_cols = ['系统负荷预测值', '风光总加预测值', '光伏预测值',
                '水电预测值', '非市场化机组预测值']
time_features = ['hour', 'month']
target_col = 'A'


def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['month'] = df['times'].dt.month
    return df


if __name__ == '__main__':
    print("=" * 60)
    print("v12 - Cross Validation + Residual Analysis")
    print("=" * 60)

    # ==================== 数据准备 ====================
    print("\n[1/4] Loading data...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)

    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    df_train = add_time_features(df_train)

    all_features = feature_cols + time_features
    X = df_train[all_features].values
    y = df_train[target_col].values

    print(f"  Total samples: {len(X)}, Features: {len(all_features)}")

    # ==================== 1. 5折时间序列交叉验证 ====================
    print("\n[2/4] 5-Fold Time Series Cross-Validation...")
    print("  (Using TimeSeriesSplit to preserve temporal order)")

    # 时间序列分割：不能用随机KFold，必须保证验证集在训练集之后
    tscv = TimeSeriesSplit(n_splits=5)

    cv_rmse_scores = []
    cv_mae_scores = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            verbose=0
        )
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        mae = mean_absolute_error(y_val, y_val_pred)

        cv_rmse_scores.append(rmse)
        cv_mae_scores.append(mae)
        fold_models.append(model)

        print(f"  Fold {fold}: RMSE={rmse:.6f}, MAE={mae:.6f}, "
              f"Train={len(train_idx)}, Val={len(val_idx)}")

    print(f"\n  CV Mean RMSE: {np.mean(cv_rmse_scores):.6f} (+/- {np.std(cv_rmse_scores):.6f})")
    print(f"  CV Mean MAE:  {np.mean(cv_mae_scores):.6f} (+/- {np.std(cv_mae_scores):.6f})")

    # ==================== 2. 残差分析 ====================
    print("\n[3/4] Residual Analysis...")

    # 使用最后一个fold的模型进行残差分析
    best_model = fold_models[-1]
    y_all_pred = best_model.predict(X)
    residuals = y - y_all_pred

    print(f"  Total residuals: {len(residuals)}")
    print(f"  Mean residual: {np.mean(residuals):.6f} (should be ~0)")
    print(f"  Std residual: {np.std(residuals):.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(residuals**2)):.6f}")

    # 残差分布
    print("\n  Residual Percentiles:")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    for p in percentiles:
        val = np.percentile(residuals, p)
        print(f"    {p:3d}%: {val:+.4f}")

    # 残差直方图（简化版，只打印统计）
    print("\n  Residual Histogram:")
    bins = [-np.inf, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, np.inf]
    labels = ['<-0.5', '-0.5~-0.25', '-0.25~-0.1', '-0.1~0.1', '0.1~0.25', '0.25~0.5', '>0.5']
    counts, _ = np.histogram(residuals, bins=bins)
    for label, count in zip(labels, counts):
        pct = count / len(residuals) * 100
        bar = "#" * int(pct / 2)
        print(f"    {label:12s}: {count:6d} ({pct:5.1f}%) {bar}")

    # ==================== 3. 残差 vs 特征 分析 ====================
    print("\n[4/4] Residual vs Features Analysis...")
    print("  (Checking if residuals correlate with any feature - indicates missing signal)")

    residual_df = pd.DataFrame({'residual': residuals})
    for i, feat in enumerate(all_features):
        residual_df[feat] = X[:, i]

    correlations = {}
    for feat in all_features:
        corr = np.corrcoef(residual_df[feat], residual_df['residual'])[0, 1]
        correlations[feat] = corr

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\n  Residual correlation with features:")
    for feat, corr in sorted_corr:
        indicator = ""
        if abs(corr) > 0.1:
            indicator = " <-- MISSING SIGNAL!"
        print(f"    {feat:<20s}: {corr:+.4f}{indicator}")

    # ==================== 4. 时间模式残差 ====================
    print("\n  Residual by hour:")
    df_analysis = df_train.copy()
    df_analysis['residual'] = residuals
    hourly_residuals = df_analysis.groupby('hour')['residual'].agg(['mean', 'std'])
    for hour in range(24):
        if hour in hourly_residuals.index:
            mean_val = hourly_residuals.loc[hour, 'mean']
            std_val = hourly_residuals.loc[hour, 'std']
            bar = "=" * int(abs(mean_val) * 50)
            sign = "+" if mean_val > 0 else ""
            print(f"    Hour {hour:02d}: {sign}{mean_val:.4f} (std={std_val:.4f}) {bar}")

    # ==================== 结论 ====================
    print("\n" + "=" * 60)
    print("Analysis Conclusions")
    print("=" * 60)

    # 检查是否有系统性的残差偏差
    if abs(np.mean(residuals)) > 0.01:
        print(f"\n[!] Residual mean is non-zero: {np.mean(residuals):.6f}")
        print("    Model may have systematic bias")
    else:
        print("\n[OK] Residual mean is ~0, no systematic bias")

    # 检查是否有高相关残差
    high_corr_residuals = [(f, c) for f, c in sorted_corr if abs(c) > 0.1]
    if high_corr_residuals:
        print("\n[!] Features with high residual correlation:")
        for feat, corr in high_corr_residuals:
            print(f"    {feat}: r={corr:.4f} - consider adding interaction")
    else:
        print("\n[OK] No strong residual correlations found")

    print("\n" + "=" * 60)