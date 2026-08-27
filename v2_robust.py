"""
储能优化策略 v2: 预测误差感知策略
核心改进: 根据预测误差分布调整充放电功率，降低不确定性风险
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 路径配置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'v2_robust_price.csv')
output_power_path = os.path.join(output_dir, 'v2_robust_output.csv')

feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'

# v2新增: 风险调节参数
RISK_ALPHA = 0.7  # 风险厌恶程度 (0-1, 越大越保守)
POWER_SCALE_MIN = 0.5  # 最小功率缩放

def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df

def compute_error_distribution(y_true, y_pred):
    """计算预测误差的分布特征"""
    errors = np.abs(y_true - y_pred)
    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'p50': np.percentile(errors, 50),
        'p90': np.percentile(errors, 90),
        'p95': np.percentile(errors, 95)
    }

def get_power_scale(error, error_stats):
    """根据预测误差确定功率缩放因子"""
    # 误差越大，功率越保守
    if error <= error_stats['p50']:
        return 1.0
    elif error <= error_stats['p90']:
        scale = 1.0 - (error - error_stats['p50']) / (error_stats['p90'] - error_stats['p50']) * (1 - POWER_SCALE_MIN)
        return max(scale, POWER_SCALE_MIN)
    else:
        return POWER_SCALE_MIN

def generate_strategy_v2(price_csv, save_path, error_stats=None):
    """v2鲁棒策略: 根据预测误差调整功率"""
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])
    df['date'] = df['times'].dt.date

    results = []
    total_profit = 0

    for date, group in df.groupby('date'):
        prices = group['A'].values
        times = group['times'].values
        n = len(prices)
        if n != 96:
            print(f"警告: {date} 数据点={n}, 预期=96")
            continue

        best_profit = 0
        best_tc = -1
        best_td = -1
        best_charge_dur = 8
        best_discharge_dur = 8

        # 使用v0类似的固定8时段搜索，但记录最优
        for tc in range(0, 81):
            charge_prices = prices[tc:tc+8]
            charge_cost = np.sum(charge_prices) * 1000

            for td in range(tc + 8, 89):
                discharge_prices = prices[td:td+8]
                discharge_revenue = np.sum(discharge_prices) * 1000
                profit = discharge_revenue - charge_cost

                if profit > best_profit:
                    best_profit = profit
                    best_tc = tc
                    best_td = td

        power = np.zeros(96)
        if best_tc >= 0 and best_td >= 0:
            # 根据误差调整功率
            if error_stats is not None:
                # 计算该时段的平均预测不确定性
                uncertainty = (error_stats['mean_error'] + error_stats['std_error']) / 2
                power_scale = get_power_scale(uncertainty, error_stats)
            else:
                power_scale = 1.0

            power[best_tc:best_tc+8] = -1000 * power_scale
            power[best_td:best_td+8] = 1000 * power_scale
            total_profit += best_profit * power_scale
            print(f"{date}: tc={best_tc}, td={best_td}, scale={power_scale:.2f}, profit={best_profit*power_scale:.2f}")
        else:
            print(f"{date}: 无交易")

        for i, (t, p, pr) in enumerate(zip(times, power, prices)):
            results.append({'times': t, '实时价格': pr, 'power': p})

    df_result = pd.DataFrame(results)
    df_result.to_csv(save_path, index=False)

    n_days = len(df.groupby("date"))
    avg_profit = total_profit / n_days if n_days > 0 else 0
    print(f'v2鲁棒策略 - 总天数:{n_days}, 总收益:{total_profit:.2f}, 日均收益:{avg_profit:.2f}')
    return avg_profit

if __name__ == '__main__':
    print("=" * 60)
    print("v2 预测误差感知策略 - 鲁棒功率调整")
    print("=" * 60)

    # 数据加载
    print("\n[1/4] 加载数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)
    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    df_train = add_time_features(df_train)
    all_features = feature_cols + ['hour', 'minute', 'dayofweek', 'month']

    X = df_train[all_features].values
    y = df_train[target_col].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # 模型训练
    print("\n[2/4] 训练模型...")
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, verbose=0
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    print(f"  验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # 计算误差分布
    error_stats = compute_error_distribution(y_val, y_val_pred)
    print(f"  预测误差 - mean:{error_stats['mean_error']:.4f}, std:{error_stats['std_error']:.4f}")
    print(f"  预测误差 - p90:{error_stats['p90']:.4f}, p95:{error_stats['p95']:.4f}")

    # 测试集推理
    print("\n[3/4] 测试集推理...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_time_features(df_test)
    X_test = df_test[all_features].values
    y_test_pred = model.predict(X_test)

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)

    # 生成策略
    print("\n[4/4] 生成充放电策略...")
    avg_profit = generate_strategy_v2(output_price_path, output_power_path, error_stats)

    print(f"\nv2完成! 提交文件: {output_power_path}")
    print(f"日均收益: {avg_profit:.2f}")