"""
v11 基于共线性分析去除冗余特征
移除 风电预测值（与风光总加预测值 高相关，且重要性更低）
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')

output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'sklearn_baseline_output.csv')
output_power_path = os.path.join(output_dir, 'output.csv')

# v11: 基于共线性分析，去除冗余特征
# 原始特征: 系统负荷, 风光总加, 联络线, 风电, 光伏, 水电, 非市场化, hour, month
# v9: 移除 联络线预测值, dayofweek, minute
# v11: 再移除 风电预测值 (与风光总加 r=0.74，高相关，且光太阳能重要性更高)
feature_cols = ['系统负荷预测值', '风光总加预测值', '光伏预测值',
                '水电预测值', '非市场化机组预测值']
time_features = ['hour', 'month']
target_col = 'A'


def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


def generate_strategy(price_csv, save_path):
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])
    df['date'] = df['times'].dt.date
    results = []
    total_profit = 0

    for date, group in df.groupby('date'):
        prices = group['A'].values
        times = group['times'].values
        if len(prices) != 96:
            continue

        best_profit, best_tc, best_td = 0, -1, -1
        for tc in range(0, 81):
            charge_cost = np.sum(prices[tc:tc+8]) * 1000
            for td in range(tc + 8, 89):
                profit = np.sum(prices[td:td+8]) * 1000 - charge_cost
                if profit > best_profit:
                    best_profit, best_tc, best_td = profit, tc, td

        power = np.zeros(96)
        if best_tc >= 0:
            power[best_tc:best_tc+8] = -1000
            power[best_td:best_td+8] = 1000
            total_profit += best_profit
        results.extend([{'times': t, '实时价格': p, 'power': pw}
                       for t, pw, p in zip(times, power, prices)])

    pd.DataFrame(results).to_csv(save_path, index=False)
    return total_profit


if __name__ == '__main__':
    print("=" * 60)
    print("v11 - Multicollinearity Based Feature Removal")
    print("=" * 60)

    # ==================== 数据准备 ====================
    print("\n[1/4] Loading data...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)

    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    df_train = add_time_features(df_train)

    # v11: 7个特征（去除风电预测值）
    all_features = feature_cols + time_features
    print(f"  [v11] Using {len(all_features)} features:")
    print(f"       Removed: 风电预测值 (r=0.74 with 风光总加)")

    X = df_train[all_features].values
    y = df_train[target_col].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # ==================== 模型训练 ====================
    print("\n[2/4] Training model...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        verbose=1
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    print(f'\n  Validation RMSE: {rmse:.6f}, MAE: {mae:.6f}')

    # ==================== 测试推理 ====================
    print("\n[3/4] Test inference...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_time_features(df_test)

    X_test = df_test[all_features].values
    y_test_pred = model.predict(X_test)

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)
    print(f'  Output: {output_price_path}')

    # ==================== 生成策略 ====================
    print("\n[4/4] Generating strategy...")
    total_profit = generate_strategy(output_price_path, output_power_path)
    n_days = len(df_out) // 96
    print(f'  Total profit: {total_profit:.2f}, Avg: {total_profit/n_days:.2f}')

    # 特征重要性
    print("\n" + "=" * 60)
    print("Feature Importance (v11)")
    print("=" * 60)
    for feat, imp in sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<20s}: {imp:.4f}")

    print("\n" + "=" * 60)
    print("v11 completed!")
    print("=" * 60)
