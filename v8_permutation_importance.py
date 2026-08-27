"""
排列重要性(Permutation Importance)分析
比GDBT内置的feature_importances_更可靠
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ==================== 路径配置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')

feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'


def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


if __name__ == '__main__':
    print("=" * 60)
    print("Permutation Importance Analysis")
    print("=" * 60)

    # ==================== 数据准备 ====================
    print("\n[1/3] Loading data...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)

    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    df_train = add_time_features(df_train)

    # 测试11个特征（原始）
    all_features = feature_cols + ['hour', 'minute', 'dayofweek', 'month']

    X = df_train[all_features].values
    y = df_train[target_col].values

    # 按时间顺序划分
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # ==================== 模型训练 ====================
    print("\n[2/3] Training model...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        verbose=0
    )
    model.fit(X_train, y_train)

    # 基础RMSE
    y_val_pred = model.predict(X_val)
    base_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"  Base RMSE: {base_rmse:.6f}")

    # ==================== 排列重要性计算 ====================
    print("\n[3/3] Computing Permutation Importance...")
    from sklearn.inspection import permutation_importance

    # n_repeats=30 表示每个特征打乱30次取平均，更稳定
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=30,
        random_state=42,
        n_jobs=-1
    )

    # 整理结果
    perm_importance = result.importances_mean
    perm_std = result.importances_std

    # 与原始feature_importances对比
    builtin_importance = model.feature_importances_

    # 按排列重要性排序
    sorted_idx = perm_importance.argsort()[::-1]

    print("\n" + "=" * 80)
    print(f"{'Feature':<20} {'Perm_Imp':>10} {'+/─':>8} {'Built-in':>10} {'Diff':>10}")
    print("=" * 80)

    for idx in sorted_idx:
        feat = all_features[idx]
        perm = perm_importance[idx]
        std = perm_std[idx]
        builtin = builtin_importance[idx]
        diff = perm - builtin
        print(f"{feat:<20} {perm:>10.4f} {std:>8.4f} {builtin:>10.4f} {diff:>+10.4f}")

    # ==================== 关键发现 ====================
    print("\n" + "=" * 60)
    print("Key Findings")
    print("=" * 60)

    # 找出排列重要性为负的特征（打乱后RMSE反而下降=噪声特征）
    negative_imp = [(all_features[i], perm_importance[i])
                     for i in range(len(all_features)) if perm_importance[i] < 0]

    if negative_imp:
        print("\n[!] Negative importance features (may be noise):")
        for feat, imp in negative_imp:
            print(f"    {feat}: {imp:.4f}")
    else:
        print("\n[OK] All features have positive importance")

    # 找出两种方法差异大的特征
    print("\n[!] Features with large discrepancy (|perm - builtin| > 0.05):")
    for i, feat in enumerate(all_features):
        diff = abs(perm_importance[i] - builtin_importance[i])
        if diff > 0.05:
            print(f"    {feat}: perm={perm_importance[i]:.4f}, builtin={builtin_importance[i]:.4f}, diff={diff:.4f}")

    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)
