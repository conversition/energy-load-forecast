"""
v14 超参数调优 + 早停法
1. 用GridSearchCV调优 n_estimators, max_depth, learning_rate, subsample
2. 用早停法让验证集决定训练轮数
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')

output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'sklearn_baseline_output.csv')
output_power_path = os.path.join(output_dir, 'output.csv')

# v13的10个特征
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
    print("v14 - Hyperparameter Tuning + Early Stopping")
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

    # ==================== 1. 早停法训练 ====================
    print("\n[2/5] Training with Early Stopping...")
    print("  Monitoring validation error to stop when no improvement")

    # 记录训练历史，手动实现早停
    best_rmse = float('inf')
    best_iter = 0
    no_improve_count = 0
    train_losses = []

    model = GradientBoostingRegressor(
        n_estimators=500,  # 设置更大的上界，让早停来决定
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        verbose=0
    )

    # 分批训练，每50轮检查一次
    for i in range(10):
        start_iter = i * 50 + 1
        end_iter = (i + 1) * 50

        # 重新创建模型，使用warm_start
        if i == 0:
            model = GradientBoostingRegressor(
                n_estimators=end_iter,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                warm_start=True,
                verbose=0
            )
            model.fit(X_train, y_train)
        else:
            model.n_estimators = end_iter
            model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_losses.append((end_iter, rmse))

        if rmse < best_rmse:
            best_rmse = rmse
            best_iter = end_iter
            no_improve_count = 0
            print(f"  Iter {end_iter}: RMSE={rmse:.6f} * best")
        else:
            no_improve_count += 1
            print(f"  Iter {end_iter}: RMSE={rmse:.6f}")

        if no_improve_count >= 3:
            print(f"\n  Early stopping at iteration {end_iter}")
            print(f"  Best iteration: {best_iter}, Best RMSE: {best_rmse:.6f}")
            break

    print(f"\n  Early stopped: best_iter={best_iter}, best_rmse={best_rmse:.6f}")

    # ==================== 2. 超参数调优 ====================
    print("\n[3/5] Grid Search CV (using best iteration from early stopping)...")

    # 基于早停结果，确定搜索范围
    param_grid = {
        'n_estimators': [best_iter, best_iter + 50, best_iter + 100],
        'max_depth': [4, 5, 6, 7],
        'learning_rate': [0.03, 0.05, 0.08],
        'subsample': [0.7, 0.8, 0.9]
    }

    tscv = TimeSeriesSplit(n_splits=3)  # 用3折加速

    # 限制参数组合数量
    print(f"  Testing combinations...")
    results = []

    # 简化搜索：先调 max_depth 和 n_estimators
    for max_depth in [4, 5, 6]:
        for n_est in [best_iter, best_iter + 50, best_iter + 100]:
            if n_est < 1:
                n_est = 1  # 确保至少为1
            model = GradientBoostingRegressor(
                n_estimators=n_est,
                learning_rate=0.05,
                max_depth=max_depth,
                subsample=0.8,
                verbose=0
            )

            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_vl = X_train[train_idx], X_train[val_idx]
                y_tr, y_vl = y_train[train_idx], y_train[val_idx]
                model.fit(X_tr, y_tr)
                rmse = np.sqrt(mean_squared_error(y_vl, model.predict(X_vl)))
                cv_scores.append(rmse)

            mean_cv = np.mean(cv_scores)
            results.append({
                'max_depth': max_depth,
                'n_estimators': n_est,
                'cv_rmse': mean_cv
            })
            print(f"    max_depth={max_depth}, n_est={n_est}: CV RMSE={mean_cv:.6f}")

    # 找最佳
    best_result = min(results, key=lambda x: x['cv_rmse'])
    print(f"\n  Best from grid search: max_depth={best_result['max_depth']}, "
          f"n_est={best_result['n_estimators']}, CV RMSE={best_result['cv_rmse']:.6f}")

    # ==================== 3. 用最佳参数训练最终模型 ====================
    print("\n[4/5] Training final model with best params...")

    final_model = GradientBoostingRegressor(
        n_estimators=best_result['n_estimators'],
        learning_rate=0.05,
        max_depth=best_result['max_depth'],
        subsample=0.8,
        verbose=1
    )
    final_model.fit(X_train, y_train)

    y_val_pred = final_model.predict(X_val)
    final_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    final_mae = mean_absolute_error(y_val, y_val_pred)
    print(f'\n  Final Validation RMSE: {final_rmse:.6f}, MAE: {final_mae:.6f}')

    # ==================== 4. 测试推理 ====================
    print("\n[5/5] Test inference...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_features(df_test)

    X_test = df_test[base_features].values
    y_test_pred = final_model.predict(X_test)

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)

    total_profit = generate_strategy(output_price_path, output_power_path)
    n_days = len(df_out) // 96
    print(f'  Total profit: {total_profit:.2f}, Avg: {total_profit/n_days:.2f}')

    # 特征重要性
    print("\n" + "=" * 60)
    print("Feature Importance (v14)")
    print("=" * 60)
    for feat, imp in sorted(zip(base_features, final_model.feature_importances_), key=lambda x: -x[1]):
        bar = "=" * int(imp * 100)
        print(f"  {feat:<20s}: {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("v14 completed!")
    print("=" * 60)