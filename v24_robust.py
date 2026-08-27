"""
v24 稳健版 - 坚守v21按天架构 + HistGradientBoosting + 稳健策略
核心升级:
1. 使用HistGradientBoostingRegressor替代GBDT
2. 添加generate_robust_strategy（抗预测噪声的稳健策略）
3. 保留v21的按天递推架构
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')

output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'v24_robust_price.csv')
output_power_path = os.path.join(output_dir, 'v24_robust_output.csv')

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


def generate_robust_strategy(price_csv, save_path, min_profit_threshold=0, noise_margin=0.10):
    """
    生成稳健充放电策略 (抗预测噪声)

    :param price_csv: 预测电价数据路径
    :param save_path: 策略输出保存路径
    :param min_profit_threshold: 最坏情况下的最低保底利润阈值
    :param noise_margin: 预期最大误差幅度 (如 0.10 代表考虑 ±10% 的电价波动)
    :return: 按照预测值计算的名义总收益
    """
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])
    df['date'] = df['times'].dt.date
    results = []

    nominal_total_profit = 0
    days_operated = 0

    for date, group in df.groupby('date'):
        prices = group['A'].values
        times = group['times'].values
        if len(prices) != 96:
            continue

        best_worst_case_profit = -float('inf')
        best_tc_candidate, best_td_candidate = -1, -1
        best_nominal_profit_for_this_plan = 0

        for tc in range(0, 81):
            base_charge_cost = np.sum(prices[tc:tc+8]) * 1000
            worst_charge_cost = base_charge_cost * (1 + noise_margin)

            for td in range(tc + 8, 89):
                base_discharge_rev = np.sum(prices[td:td+8]) * 1000
                worst_discharge_rev = base_discharge_rev * (1 - noise_margin)

                worst_profit = worst_discharge_rev - worst_charge_cost

                if worst_profit > best_worst_case_profit:
                    best_worst_case_profit = worst_profit
                    best_tc_candidate = tc
                    best_td_candidate = td
                    best_nominal_profit_for_this_plan = base_discharge_rev - base_charge_cost

        power = np.zeros(96)
        if best_tc_candidate >= 0 and best_worst_case_profit >= min_profit_threshold:
            power[best_tc_candidate:best_tc_candidate+8] = -1000
            power[best_td_candidate:best_td_candidate+8] = 1000
            nominal_total_profit += best_nominal_profit_for_this_plan
            days_operated += 1

        results.extend([{'times': t, '实时价格': p, 'power': pw}
                       for t, pw, p in zip(times, power, prices)])

    pd.DataFrame(results).to_csv(save_path, index=False)

    total_days = len(df['date'].unique())
    print(f"  策略执行: 容忍度 ±{noise_margin*100}%, 保底阈值 {min_profit_threshold}")
    print(f"  共 {total_days} 天, 稳健操作 {days_operated} 天 (规避了 {total_days - days_operated} 天高风险交易)")

    return nominal_total_profit


if __name__ == '__main__':
    print("=" * 60)
    print("v24 稳健版 - HistGradientBoosting + 稳健策略")
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
    print("\n[2/5] 训练 HistGradientBoostingRegressor 模型...")
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_depth=7,
        max_leaf_nodes=63,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.1
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
    print("\n[4/5] 递推预测测试集 (按天架构)...")

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

    # ==================== 5. 稳健策略网格搜索 ====================
    print("\n[5/5] 开始稳健策略网格搜索...")
    df_out = pd.DataFrame({'times': df_test['times'], target_col: predicted_prices})
    df_out.to_csv(output_price_path, index=False)

    best_profit = 0
    best_params = (0, 0)

    for noise in [0.0, 0.05, 0.10, 0.15]:
        for threshold in [0, 1000, 3000, 5000]:
            profit = generate_robust_strategy(
                output_price_path,
                "temp.csv",
                min_profit_threshold=threshold,
                noise_margin=noise
            )
            print(f"  Noise: {noise:.2f}, Threshold: {threshold} -> 名义预期收益: {profit:.2f}")
            if profit > best_profit:
                best_profit = profit
                best_params = (noise, threshold)

    print(f"\n最优参数: noise={best_params[0]}, threshold={best_params[1]}, 收益: {best_profit:.2f}")

    # 用最优参数生成最终文件
    final_profit = generate_robust_strategy(
        output_price_path,
        output_power_path,
        min_profit_threshold=best_params[1],
        noise_margin=best_params[0]
    )
    print(f"最终稳健策略已保存")

    print("\n" + "=" * 60)
    print("v24 稳健版完成!")
    print("=" * 60)