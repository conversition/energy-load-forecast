"""
v23 终极突破版 - LightGBM + 逐点递推 + 多阶滞后
核心升级:
1. 引入 LightGBM 替代 GBDT，拟合更强，防过拟合更优秀
2. 推理架构升级为"逐点递推"，支持任意粒度的滞后特征
3. 新增 price_lag_1 (前一刻), price_lag_48 (半天前)
"""
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')

output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'v23_lgb_point_price.csv')
output_power_path = os.path.join(output_dir, 'v23_lgb_point_output.csv')

target_col = 'A'

LAG_1 = 1
LAG_48 = 48
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
    """添加多阶滞后特征"""
    df = df.copy()
    df['price_lag_1'] = df[price_col].shift(LAG_1)
    df['price_lag_48'] = df[price_col].shift(LAG_48)
    df['price_lag_96'] = df[price_col].shift(LAG_96)
    df['price_lag_192'] = df[price_col].shift(LAG_192)
    return df


def add_rolling_stats(df):
    """基于 price_lag_1 计算滚动统计量, 绝对避免数据穿越"""
    df = df.copy()
    rolling = df['price_lag_1'].rolling(window=LAG_96, min_periods=1)
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
    df = add_rolling_stats(df)
    df = df.iloc[LAG_192:].reset_index(drop=True)
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
    print("v23 突破版 - LightGBM + 逐点递推多滞后")
    print("=" * 60)

    # ==================== 特征定义 ====================
    base_features = [
        '系统负荷预测值', '风光总加预测值', '联络线预测值',
        '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值'
    ]
    time_features = ['hour', 'minute', 'dayofweek']
    cyclical_features = ['sin_month', 'cos_month', 'sin_hour', 'cos_hour']
    business_features = ['净负荷', '风光渗透率']
    lag_features = ['price_lag_1', 'price_lag_48', 'price_lag_96', 'price_lag_192']
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

    # ==================== 2. 模型训练 (LightGBM) ====================
    print("\n[2/5] 训练 LightGBM 模型...")
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(stopping_rounds=30)])

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

    # ==================== 4. 逐点递推预测 ====================
    print("\n[4/5] 逐点递推预测测试集 (Point-by-Point)...")

    all_predicted = df_train[target_col].values[-LAG_192:].copy()
    X_test_base_arr = df_test[base_features + time_features + cyclical_features + business_features].values
    predicted_prices = np.zeros(len(df_test))

    for i in range(len(df_test)):
        base_feats = X_test_base_arr[i]

        lag_1 = all_predicted[-LAG_1]
        lag_48 = all_predicted[-LAG_48]
        lag_96 = all_predicted[-LAG_96]
        lag_192 = all_predicted[-LAG_192]

        past_96 = all_predicted[-LAG_96:]
        roll_mean = np.mean(past_96)
        roll_std = np.std(past_96)

        lag_feats = [lag_1, lag_48, lag_96, lag_192]
        stat_feats = [roll_mean, roll_std]
        x_in = np.concatenate([base_feats, lag_feats, stat_feats]).reshape(1, -1)

        pred = model.predict(x_in)[0]
        predicted_prices[i] = pred
        all_predicted = np.append(all_predicted, pred)

        if i % 96 == 0:
            print(f"  预测到第 {i//96 + 1} 天...")

    # ==================== 5. 保存结果 ====================
    print("\n[5/5] 生成充放电策略...")
    df_out = pd.DataFrame({'times': df_test['times'], target_col: predicted_prices})
    df_out.to_csv(output_price_path, index=False)

    total_profit = generate_strategy(output_price_path, output_power_path)
    n_days = len(df_test) // 96
    print(f"\n  总收益: {total_profit:.2f}, 日均收益: {total_profit/n_days:.2f}")

    # 特征重要性
    print("\n" + "=" * 60)
    print("Feature Importance (LightGBM)")
    print("=" * 60)
    importances = model.feature_importances_
    importances = importances / importances.sum()
    for feat, imp in sorted(zip(all_features, importances), key=lambda x: -x[1]):
        bar = "=" * int(imp * 100)
        print(f"  {feat:<20s}: {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("v23 突破版完成!")
    print("=" * 60)