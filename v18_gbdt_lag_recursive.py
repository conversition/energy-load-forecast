"""
v18 GBDT + 滞后特征 + 递推预测
基于分析报告的完整修复:
1. 添加 price_lag_96 滞后特征 (训练集用shift, 测试集用递推)
2. 换回GBDT捕捉极端价格
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
output_price_path = os.path.join(output_dir, 'v18_gbdt_lag_price.csv')
output_power_path = os.path.join(output_dir, 'v18_gbdt_lag_output.csv')

target_col = 'A'

# 滞后阶数: 96点=1天(15分钟粒度)
LAG_96 = 96


def add_time_features(df):
    """添加基础时间特征"""
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


def add_cyclical_features(df):
    """月份和小时的周期编码"""
    df = df.copy()
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


def add_business_features(df):
    """业务特征: 净负荷和风光渗透率"""
    df = df.copy()
    df['净负荷'] = df['系统负荷预测值'] - df['风光总加预测值']
    total_gen = df['风光总加预测值'] + df['水电预测值'] + df['非市场化机组预测值']
    df['风光渗透率'] = df['风光总加预测值'] / (total_gen + 1)
    return df


def add_lag_features(df, price_col):
    """添加价格滞后特征

    关键: 使用shift(96)获取1天前的电价
    前96个点会是NaN，会在后续截断处理
    """
    df = df.copy()
    # shift后重命名为lag特征
    df['price_lag_96'] = df[price_col].shift(LAG_96)
    return df


def prepare_train_data(df_feat, df_label):
    """准备训练数据

    流程:
    1. 合并特征和标签
    2. 添加所有特征
    3. 添加滞后特征
    4. 截断前LAG_96个无效行
    """
    df = pd.merge(df_feat, df_label, on='times', how='inner')
    df['times'] = pd.to_datetime(df['times'])
    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_business_features(df)
    df = add_lag_features(df, target_col)

    # 截断前LAG_96个无效行
    df = df.iloc[LAG_96:].reset_index(drop=True)
    return df


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
    print("v18 GBDT + 滞后特征 + 递推预测")
    print("=" * 60)

    # ==================== 特征定义 ====================
    base_features = [
        '系统负荷预测值', '风光总加预测值', '联络线预测值',
        '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值'
    ]
    time_features = ['hour', 'minute', 'dayofweek']
    cyclical_features = ['sin_month', 'cos_month', 'sin_hour', 'cos_hour']
    business_features = ['净负荷', '风光渗透率']
    lag_features = ['price_lag_96']

    all_features = base_features + time_features + cyclical_features + business_features + lag_features

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

    # ==================== 4. 递推预测 (关键!) ====================
    print("\n[4/5] 递推预测测试集...")
    """
    递推预测逻辑:
    - 测试集第一天没有前一天的真是电价，我们用训练集最后一天的电价作为lag
    - 从第二天开始，用前一天预测出的电价填入price_lag_96
    - 这样实现真正的滚动预测
    """
    # 获取训练集最后96个点的真实电价作为初始滞后值
    train_last_prices = df_train[target_col].values[-LAG_96:]
    print(f"  初始滞后值(来自训练集末尾): shape={train_last_prices.shape}")

    # 准备测试集的基础特征(不含滞后特征)
    X_test_base = df_test[base_features + time_features + cyclical_features + business_features].copy()

    # 创建价格容器
    predicted_prices = np.zeros(len(df_test))

    # 按天递推预测 (每天96个点)
    n_days = len(df_test) // 96
    current_lag = train_last_prices.copy()

    for day_idx in range(n_days):
        start_idx = day_idx * 96
        end_idx = start_idx + 96

        # 构建当天的完整特征(含滞后特征)
        day_X = X_test_base.iloc[start_idx:end_idx].values.copy()
        lag_col_idx = all_features.index('price_lag_96')

        # 添加滞后特征: 第一天用训练集末尾, 之后用前一天预测值
        day_lag = current_lag  # 96个价格值
        day_X_with_lag = np.column_stack([day_X, day_lag])

        # 预测当天电价
        day_pred = model.predict(day_X_with_lag)
        predicted_prices[start_idx:end_idx] = day_pred

        # 更新滞后值: 用当天预测值作为下一天的滞后特征
        current_lag = day_pred.copy()

        if day_idx == 0:
            print(f"  Day {day_idx+1}: 初始滞后, 预测价格均值={day_pred.mean():.4f}")
        else:
            print(f"  Day {day_idx+1}: 使用前一天预测, 预测价格均值={day_pred.mean():.4f}")

    # ==================== 5. 保存结果 ====================
    df_out = pd.DataFrame({'times': df_test['times'], target_col: predicted_prices})
    df_out.to_csv(output_price_path, index=False)

    total_profit = generate_strategy(output_price_path, output_power_path)
    print(f"\n  总收益: {total_profit:.2f}, 日均收益: {total_profit/n_days:.2f}")

    # 特征重要性
    print("\n" + "=" * 60)
    print("Feature Importance (v18 GBDT + 滞后特征)")
    print("=" * 60)
    for feat, imp in sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1]):
        bar = "=" * int(imp * 100)
        print(f"  {feat:<20s}: {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("v18 完成!")
    print("=" * 60)