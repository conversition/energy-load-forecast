"""
v22 策略优化版 - 基于v21添加最小利润阈值过滤
核心改进: 通过min_profit_threshold过滤低质量交易日,减少错误套利
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
output_price_path = os.path.join(output_dir, 'v22_strategy_price.csv')
output_power_path = os.path.join(output_dir, 'v22_strategy_output.csv')

target_col = 'A'

LAG_96 = 96
LAG_192 = 192


def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


def add_cyclical_features(df):
    df = df.copy()
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


def add_business_features(df):
    df = df.copy()
    df['净负荷'] = df['系统负荷预测值'] - df['风光总加预测值']
    total_gen = df['风光总加预测值'] + df['水电预测值'] + df['非市场化机组预测值']
    df['风光渗透率'] = df['风光总加预测值'] / (total_gen + 1)
    return df


def add_lag_features(df, price_col):
    df = df.copy()
    df['price_lag_96'] = df[price_col].shift(LAG_96)
    df['price_lag_192'] = df[price_col].shift(LAG_192)
    return df


def add_rolling_stats(df, price_col):
    """基于price_lag_96计算滚动统计量,避免target leakage"""
    df = df.copy()
    rolling = df['price_lag_96'].rolling(window=LAG_96, min_periods=1)
    df['rolling_mean'] = rolling.mean()
    df['rolling_std'] = rolling.std()
    return df


def prepare_train_data(df_feat, df_label):
    df = pd.merge(df_feat, df_label, on='times', how='inner')
    df['times'] = pd.to_datetime(df['times'])
    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_business_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_stats(df, target_col)
    df = df.iloc[LAG_192:].reset_index(drop=True)
    return df


def generate_strategy(price_csv, save_path, min_profit_threshold=0):
    """生成充放电策略 (带阈值兜底)

    :param price_csv: 预测电价数据路径
    :param save_path: 策略输出保存路径
    :param min_profit_threshold: 最小利润阈值，低于此值则当天不操作
    :return: 总收益
    """
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])
    df['date'] = df['times'].dt.date
    results = []
    total_profit = 0
    days_operated = 0

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
        if best_tc >= 0 and best_profit >= min_profit_threshold:
            power[best_tc:best_tc+8] = -1000
            power[best_td:best_td+8] = 1000
            total_profit += best_profit
            days_operated += 1

        results.extend([{'times': t, '实时价格': p, 'power': pw}
                       for t, pw, p in zip(times, power, prices)])

    pd.DataFrame(results).to_csv(save_path, index=False)
    total_days = len(df['date'].unique())
    print(f"  策略执行: 共 {total_days} 天, 实际操作 {days_operated} 天 (过滤了 {total_days - days_operated} 天)")
    return total_profit


if __name__ == '__main__':
    print("=" * 60)
    print("v22 策略优化版 - 最小利润阈值过滤")
    print("=" * 60)

    # ==================== 特征定义 ====================
    base_features = [
        '系统负荷预测值', '风光总加预测值', '联络线预测值',
        '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值'
    ]
    time_features = ['hour', 'minute', 'dayofweek']
    cyclical_features = ['sin_month', 'cos_month', 'sin_hour', 'cos_hour']
    business_features = ['净负荷', '风光渗透率']
    lag_features = ['price_lag_96', 'price_lag_192']
    stat_features = ['rolling_mean', 'rolling_std']

    all_features = base_features + time_features + cyclical_features + business_features + lag_features + stat_features

    # ==================== 1. 训练数据准备 ====================
    print("\n[1/5] 加载并准备训练数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)

    df_train = prepare_train_data(df_feat, df_label)
    print(f"  训练样本数: {len(df_train)}")

    X = df_train[all_features].values
    y = df_train[target_col].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # ==================== 2. 模型训练 ====================
    print("\n[2/5] 训练GBDT模型...")
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
    print(f"\n  验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # ==================== 3. 加载测试数据 ====================
    print("\n[3/5] 加载测试特征数据...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_time_features(df_test)
    df_test = add_cyclical_features(df_test)
    df_test = add_business_features(df_test)
    print(f"  测试集样本数: {len(df_test)}")

    # ==================== 4. 递推预测 ====================
    print("\n[4/5] 递推预测测试集...")

    train_last_prices = df_train[target_col].values[-LAG_192:]
    print(f"  初始滞后值: shape={train_last_prices.shape}")

    X_test_base = df_test[base_features + time_features + cyclical_features + business_features].copy()

    predicted_prices = np.zeros(len(df_test))
    all_predicted = np.concatenate([train_last_prices.copy()])

    n_days = len(df_test) // 96

    for day_idx in range(n_days):
        start_idx = day_idx * 96
        end_idx = start_idx + 96

        day_X = X_test_base.iloc[start_idx:end_idx].values.copy()

        lag_96_arr = all_predicted[-LAG_96:]
        lag_192_arr = all_predicted[-LAG_192:-LAG_96]

        past_192 = all_predicted[-LAG_192:]
        rolling_mean_arr = np.array([np.mean(past_192[i:i+LAG_96]) for i in range(96)])
        rolling_std_arr = np.array([np.std(past_192[i:i+LAG_96]) for i in range(96)])

        lag_features_arr = np.column_stack([lag_96_arr, lag_192_arr])
        stat_arr = np.column_stack([rolling_mean_arr, rolling_std_arr])

        day_X_with_lag = np.concatenate([day_X, lag_features_arr, stat_arr], axis=1)

        day_pred = model.predict(day_X_with_lag)
        predicted_prices[start_idx:end_idx] = day_pred
        all_predicted = np.concatenate([all_predicted, day_pred])

        if day_idx < 5 or day_idx == n_days - 1:
            print(f"  Day {day_idx+1}: mean={day_pred.mean():.4f}")

    # ==================== 5. 策略阈值优化 ====================
    print("\n[5/5] 开始优化策略阈值...")
    df_out = pd.DataFrame({'times': df_test['times'], target_col: predicted_prices})
    df_out.to_csv(output_price_path, index=False)

    print("\n阈值网格搜索:")
    best_strategy_profit = 0
    best_threshold = 0

    for threshold in range(0, 15001, 1000):
        profit = generate_strategy(output_price_path, "temp.csv", min_profit_threshold=threshold)
        print(f"  阈值: {threshold:5d} -> 总收益: {profit:10.2f}")
        if profit > best_strategy_profit:
            best_strategy_profit = profit
            best_threshold = threshold

    print(f"\n最优阈值: {best_threshold}, 对应总收益: {best_strategy_profit:.2f}")

    final_profit = generate_strategy(output_price_path, output_power_path, min_profit_threshold=best_threshold)
    print(f"最终策略已保存，总收益: {final_profit:.2f}")

    print("\n" + "=" * 60)
    print("v22 策略优化版完成!")
    print("=" * 60)