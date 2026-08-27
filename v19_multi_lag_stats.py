"""
v19 GBDT + 多滞后特征 + 滚动统计量
基于分析报告的进一步优化:
1. 添加多滞后: price_lag_96, price_lag_192, price_lag_672 (1天/2天/7天前)
2. 添加滚动统计量: 均值、标准差、最大最小值
3. 恢复完整特征 + 周期编码 + 净负荷
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
output_price_path = os.path.join(output_dir, 'v19_multi_lag_price.csv')
output_power_path = os.path.join(output_dir, 'v19_multi_lag_output.csv')

target_col = 'A'

LAG_96 = 96
LAG_192 = 192
LAG_672 = 672


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


def add_multi_lag_features(df, price_col):
    """添加多滞后特征: 96(1天), 192(2天), 672(7天)"""
    df = df.copy()
    df['price_lag_96'] = df[price_col].shift(LAG_96)
    df['price_lag_192'] = df[price_col].shift(LAG_192)
    df['price_lag_672'] = df[price_col].shift(LAG_672)
    return df


def add_rolling_stats(df, price_col):
    """添加滚动统计量 (基于96点窗口)"""
    df = df.copy()
    rolling = df[price_col].rolling(window=LAG_96, min_periods=1)
    df['rolling_mean'] = rolling.mean()
    df['rolling_std'] = rolling.std()
    df['rolling_max'] = rolling.max()
    df['rolling_min'] = rolling.min()
    return df


def prepare_train_data(df_feat, df_label):
    """准备训练数据"""
    df = pd.merge(df_feat, df_label, on='times', how='inner')
    df['times'] = pd.to_datetime(df['times'])
    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_business_features(df)
    df = add_multi_lag_features(df, target_col)
    df = add_rolling_stats(df, target_col)
    # 截断前672个无效行
    df = df.iloc[LAG_672:].reset_index(drop=True)
    return df


def compute_rolling_stats(price_window):
    """计算滚动统计量

    对过去24小时(96点)的价格计算统计量
    """
    return {
        'rolling_mean': np.mean(price_window),
        'rolling_std': np.std(price_window),
        'rolling_max': np.max(price_window),
        'rolling_min': np.min(price_window)
    }


def generate_strategy(price_csv, save_path):
    """生成充放电策略"""
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
    print("v19 GBDT + 多滞后特征 + 滚动统计量")
    print("=" * 60)

    # ==================== 特征定义 ====================
    base_features = [
        '系统负荷预测值', '风光总加预测值', '联络线预测值',
        '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值'
    ]
    time_features = ['hour', 'minute', 'dayofweek']
    cyclical_features = ['sin_month', 'cos_month', 'sin_hour', 'cos_hour']
    business_features = ['净负荷', '风光渗透率']
    lag_features = ['price_lag_96', 'price_lag_192', 'price_lag_672']
    stat_features = ['rolling_mean', 'rolling_std', 'rolling_max', 'rolling_min']

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

    # 获取训练集最后672个点的真实电价作为初始滞后值
    train_last_prices = df_train[target_col].values[-LAG_672:]
    print(f"  初始滞后值: shape={train_last_prices.shape}")

    # 准备测试集的基础特征
    X_test_base = df_test[base_features + time_features + cyclical_features + business_features].copy()

    # 创建价格容器和完整预测历史(用于计算统计量)
    predicted_prices = np.zeros(len(df_test))
    all_predicted = np.concatenate([train_last_prices.copy()])

    n_days = len(df_test) // 96

    for day_idx in range(n_days):
        start_idx = day_idx * 96
        end_idx = start_idx + 96

        # 构建当天的基础特征
        day_X = X_test_base.iloc[start_idx:end_idx].values.copy()

        # 获取多滞后特征 (使用均值作为聚合值)
        lag_96_mean = np.mean(all_predicted[-LAG_96:])
        lag_192_mean = np.mean(all_predicted[-LAG_192:])
        lag_672_mean = np.mean(all_predicted[-LAG_672:])

        # 构建滞后特征列 (96行 x 3列)
        lag_features_arr = np.column_stack([
            np.full(96, lag_96_mean),
            np.full(96, lag_192_mean),
            np.full(96, lag_672_mean)
        ])

        # 计算滚动统计量 (基于最近96点)
        stats = compute_rolling_stats(all_predicted[-LAG_96:])
        stat_features_arr = np.column_stack([
            np.full(96, stats['rolling_mean']),
            np.full(96, stats['rolling_std']),
            np.full(96, stats['rolling_max']),
            np.full(96, stats['rolling_min'])
        ])

        # 拼接所有特征
        day_X_with_lag = np.concatenate([day_X, lag_features_arr, stat_features_arr], axis=1)

        # 预测当天电价
        day_pred = model.predict(day_X_with_lag)
        predicted_prices[start_idx:end_idx] = day_pred

        # 更新预测历史
        all_predicted = np.concatenate([all_predicted, day_pred])

        if day_idx < 5 or day_idx == n_days - 1:
            print(f"  Day {day_idx+1}: mean={day_pred.mean():.4f}, "
                  f"rolling_mean={stats['rolling_mean']:.4f}, "
                  f"rolling_std={stats['rolling_std']:.4f}")

    # ==================== 5. 保存结果 ====================
    df_out = pd.DataFrame({'times': df_test['times'], target_col: predicted_prices})
    df_out.to_csv(output_price_path, index=False)

    total_profit = generate_strategy(output_price_path, output_power_path)
    print(f"\n  总收益: {total_profit:.2f}, 日均收益: {total_profit/n_days:.2f}")

    # 特征重要性
    print("\n" + "=" * 60)
    print("Feature Importance (v19 多滞后+滚动统计)")
    print("=" * 60)
    for feat, imp in sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1]):
        bar = "=" * int(imp * 100)
        print(f"  {feat:<20s}: {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("v19 完成!")
    print("=" * 60)