"""
v16 ExtraTrees 参数调优
基于v15发现：ExtraTrees单模型收益最高(768,231)，进一步调优
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')

output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'sklearn_baseline_output.csv')
output_power_path = os.path.join(output_dir, 'output.csv')

# v13 特征
base_features = ['系统负荷预测值', '风光总加预测值', '光伏预测值',
                 '水电预测值', '非市场化机组预测值', 'hour', 'month',
                 '负荷率', '季节_负荷交互', '光伏季节性']
target_col = 'A'


def add_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['month'] = df['times'].dt.month
    df['负荷率'] = df['系统负荷预测值'] / (df['风光总加预测值'] + 1)
    df['季节_负荷交互'] = df['month'] * df['系统负荷预测值']
    df['光伏季节性'] = df['光伏预测值'] * np.sin(2 * np.pi * df['month'] / 12)
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
    print("v16 - ExtraTrees Hyperparameter Tuning")
    print("=" * 60)

    # ==================== 数据准备 ====================
    print("\n[1/5] Loading data...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)

    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    df_train = add_features(df_train)

    X = df_train[base_features].values
    y = df_train[target_col].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # ==================== 1. 树数量 vs 性能 ====================
    print("\n[2/5] Tuning n_estimators...")
    tscv = TimeSeriesSplit(n_splits=3)

    n_est_results = []
    for n_est in [100, 200, 300, 400, 500]:
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            model = ExtraTreesRegressor(
                n_estimators=n_est, max_depth=15, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            )
            model.fit(X_train[train_idx], y_train[train_idx])
            pred = model.predict(X_train[val_idx])
            cv_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], pred)))

        mean_cv = np.mean(cv_scores)
        n_est_results.append((n_est, mean_cv))
        print(f"  n_estimators={n_est}: CV RMSE={mean_cv:.6f}")

    best_n_est = min(n_est_results, key=lambda x: x[1])[0]
    print(f"\n  Best n_estimators: {best_n_est}")

    # ==================== 2. max_depth 调优 ====================
    print("\n[3/5] Tuning max_depth...")
    depth_results = []

    for max_depth in [10, 12, 15, 18, 20, None]:
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            model = ExtraTreesRegressor(
                n_estimators=best_n_est, max_depth=max_depth, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            )
            model.fit(X_train[train_idx], y_train[train_idx])
            pred = model.predict(X_train[val_idx])
            cv_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], pred)))

        mean_cv = np.mean(cv_scores)
        depth_str = str(max_depth) if max_depth else "None"
        depth_results.append((max_depth, mean_cv))
        print(f"  max_depth={depth_str:5s}: CV RMSE={mean_cv:.6f}")

    best_depth = min(depth_results, key=lambda x: x[1])[0]
    print(f"\n  Best max_depth: {best_depth}")

    # ==================== 3. min_samples_leaf 调优 ====================
    print("\n[4/5] Tuning min_samples_leaf...")
    leaf_results = []

    for min_leaf in [2, 5, 10, 20, 50]:
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            model = ExtraTreesRegressor(
                n_estimators=best_n_est, max_depth=best_depth, min_samples_leaf=min_leaf,
                n_jobs=-1, random_state=42
            )
            model.fit(X_train[train_idx], y_train[train_idx])
            pred = model.predict(X_train[val_idx])
            cv_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], pred)))

        mean_cv = np.mean(cv_scores)
        leaf_results.append((min_leaf, mean_cv))
        print(f"  min_samples_leaf={min_leaf:3d}: CV RMSE={mean_cv:.6f}")

    best_leaf = min(leaf_results, key=lambda x: x[1])[0]
    print(f"\n  Best min_samples_leaf: {best_leaf}")

    # ==================== 4. 最终模型训练 ====================
    print("\n[5/5] Training final model...")
    print(f"  Best params: n_estimators={best_n_est}, max_depth={best_depth}, min_samples_leaf={best_leaf}")

    final_model = ExtraTreesRegressor(
        n_estimators=best_n_est,
        max_depth=best_depth,
        min_samples_leaf=best_leaf,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    final_model.fit(X_train, y_train)

    y_val_pred = final_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    print(f'\n  Final Val RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}')

    # ==================== 5. 测试推理 ====================
    print("\n[Test inference...]")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_features(df_test)

    X_test = df_test[base_features].values
    y_test_pred = final_model.predict(X_test)

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)

    total_profit = generate_strategy(output_price_path, output_power_path)
    n_days = len(df_out) // 96
    print(f'\n  Final Total profit: {total_profit:.2f}, Avg: {total_profit/n_days:.2f}')

    # 特征重要性
    print("\n" + "=" * 60)
    print("Feature Importance (v16)")
    print("=" * 60)
    for feat, imp in sorted(zip(base_features, final_model.feature_importances_), key=lambda x: -x[1]):
        bar = "=" * int(imp * 100)
        print(f"  {feat:<20s}: {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("v16 completed!")
    print("=" * 60)