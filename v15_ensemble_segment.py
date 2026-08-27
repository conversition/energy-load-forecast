"""
v15 异质模型集成 + 分段模型分析
1. 训练 RandomForest, ExtraTrees, GradientBoosting 并加权融合
2. 分析不同价格区间的预测表现
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
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
    print("v15 - Heterogeneous Ensemble + Segment Analysis")
    print("=" * 60)

    # ==================== 数据准备 ====================
    print("\n[1/6] Loading data...")
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

    # ==================== 1. 各模型独立训练 ====================
    print("\n[2/6] Training individual models...")

    models = {}

    # GBDT (与v13相同)
    print("  Training GradientBoosting...")
    gb = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, verbose=0
    )
    gb.fit(X_train, y_train)
    models['GBDT'] = gb
    gb_pred = gb.predict(X_val)
    gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
    print(f"    GBDT Val RMSE: {gb_rmse:.6f}")

    # RandomForest
    print("  Training RandomForest...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, verbose=0
    )
    rf.fit(X_train, y_train)
    models['RF'] = rf
    rf_pred = rf.predict(X_val)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
    print(f"    RF Val RMSE: {rf_rmse:.6f}")

    # ExtraTrees
    print("  Training ExtraTrees...")
    et = ExtraTreesRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, verbose=0
    )
    et.fit(X_train, y_train)
    models['ET'] = et
    et_pred = et.predict(X_val)
    et_rmse = np.sqrt(mean_squared_error(y_val, et_pred))
    print(f"    ET Val RMSE: {et_rmse:.6f}")

    # ==================== 2. 集成权重优化 ====================
    print("\n[3/6] Optimizing ensemble weights...")

    # 网格搜索最优权重
    best_rmse = float('inf')
    best_weights = (1/3, 1/3, 1/3)

    for w_gb in np.arange(0.2, 0.8, 0.1):
        for w_rf in np.arange(0.1, 0.6, 0.1):
            w_et = 1 - w_gb - w_rf
            if w_et < 0.1:
                continue

            ensemble_pred = w_gb * gb_pred + w_rf * rf_pred + w_et * et_pred
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))

            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = (w_gb, w_rf, w_et)

    print(f"  Best weights: GBDT={best_weights[0]:.1f}, RF={best_weights[1]:.1f}, ET={best_weights[2]:.1f}")
    print(f"  Ensemble Val RMSE: {best_rmse:.6f}")

    # 验证集集成预测
    ensemble_pred = best_weights[0] * gb_pred + best_weights[1] * rf_pred + best_weights[2] * et_pred

    # ==================== 3. 分段模型分析 ====================
    print("\n[4/6] Segment Analysis (price区间 vs 预测误差)...")

    # 按真实价格分组
    price_bins = [0, 0.5, 1.0, 1.5, 2.0, np.inf]
    bin_labels = ['<0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '>2.0']

    df_analysis = pd.DataFrame({
        'y_true': y_val,
        'y_pred_gb': gb_pred,
        'y_pred_rf': rf_pred,
        'y_pred_et': et_pred,
        'y_pred_ens': ensemble_pred
    })

    print("\n  RMSE by price segment:")
    print(f"  {'Segment':<12} {'Count':>8} {'GBDT':>10} {'RF':>10} {'ET':>10} {'Ensemble':>10}")
    print("  " + "-" * 62)

    segment_results = []
    for i, (low, high) in enumerate(zip(price_bins[:-1], price_bins[1:])):
        mask = (df_analysis['y_true'] >= low) & (df_analysis['y_true'] < high)
        count = mask.sum()
        if count > 0:
            segment_data = df_analysis[mask]

            gb_seg_rmse = np.sqrt(mean_squared_error(segment_data['y_true'], segment_data['y_pred_gb']))
            rf_seg_rmse = np.sqrt(mean_squared_error(segment_data['y_true'], segment_data['y_pred_rf']))
            et_seg_rmse = np.sqrt(mean_squared_error(segment_data['y_true'], segment_data['y_pred_et']))
            ens_seg_rmse = np.sqrt(mean_squared_error(segment_data['y_true'], segment_data['y_pred_ens']))

            label = bin_labels[i]
            print(f"  {label:<12} {count:>8} {gb_seg_rmse:>10.4f} {rf_seg_rmse:>10.4f} {et_seg_rmse:>10.4f} {ens_seg_rmse:>10.4f}")

            segment_results.append({
                'segment': label, 'count': count,
                'gb_rmse': gb_seg_rmse, 'rf_rmse': rf_seg_rmse,
                'et_rmse': et_seg_rmse, 'ens_rmse': ens_seg_rmse
            })

    # ==================== 4. 分段集成策略 ====================
    print("\n[5/6] Segment-based ensemble weights...")

    # 根据各区间表现，给不同模型分配不同权重
    # 例如：如果RF在低价区间表现更好，则在低价区间多用RF
    for seg in segment_results:
        print(f"  {seg['segment']}: ", end="")
        rmse_scores = {'GBDT': seg['gb_rmse'], 'RF': seg['rf_rmse'], 'ET': seg['et_rmse']}
        # 归一化权重（误差大的权重小）
        inv_rmse = {k: 1/v for k, v in rmse_scores.items()}
        total = sum(inv_rmse.values())
        weights = {k: v/total for k, v in inv_rmse.items()}
        print(f"GBDT={weights['GBDT']:.2f}, RF={weights['RF']:.2f}, ET={weights['ET']:.2f}")

    # ==================== 5. 最终测试推理 ====================
    print("\n[6/6] Final test inference...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_features(df_test)

    X_test = df_test[base_features].values

    # 各模型预测
    gb_test = gb.predict(X_test)
    rf_test = rf.predict(X_test)
    et_test = et.predict(X_test)

    # 加权融合
    y_test_pred = best_weights[0] * gb_test + best_weights[1] * rf_test + best_weights[2] * et_test

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)

    total_profit = generate_strategy(output_price_path, output_power_path)
    n_days = len(df_out) // 96
    print(f'\n  Ensemble Total profit: {total_profit:.2f}, Avg: {total_profit/n_days:.2f}')

    # 逐个模型单独测试
    print("\n  Individual model profits:")
    for name, pred in [('GBDT', gb_test), ('RF', rf_test), ('ET', et_test)]:
        df_single = pd.DataFrame({'times': df_test['times'], target_col: pred})
        df_single.to_csv(output_price_path, index=False)
        profit = generate_strategy(output_price_path, output_power_path)
        print(f"    {name}: {profit:.2f}")

    print("\n" + "=" * 60)
    print("v15 completed!")
    print("=" * 60)