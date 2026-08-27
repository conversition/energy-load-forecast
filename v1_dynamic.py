"""
储能优化策略 v1: 多时段动态策略
核心改进: 可变充放电时长(4-12时间点)，允许日内多轮充放电
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
output_price_path = os.path.join(output_dir, 'v1_dynamic_price.csv')
output_power_path = os.path.join(output_dir, 'v1_dynamic_output.csv')

feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'

# v1新增: 可变时长范围
MIN_DURATION = 4
MAX_DURATION = 12

def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df

def generate_strategy_v1(price_csv, save_path):
    """v1动态策略: 可变时长充放电，允许多轮"""
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
        best_plan = None  # (tc, td, charge_dur, discharge_dur)

        # 遍历所有可能的充电时长和放电时长组合
        for charge_dur in range(MIN_DURATION, MAX_DURATION + 1):
            for discharge_dur in range(MIN_DURATION, MAX_DURATION + 1):
                # 遍历所有可能的充电开始时间
                for tc in range(0, 97 - charge_dur):
                    charge_prices = prices[tc:tc+charge_dur]
                    charge_cost = np.sum(charge_prices) * 1000

                    # 放电开始时间需在充电结束后
                    for td in range(tc + charge_dur, 97 - discharge_dur):
                        discharge_prices = prices[td:td+discharge_dur]
                        discharge_revenue = np.sum(discharge_prices) * 1000
                        profit = discharge_revenue - charge_cost

                        if profit > best_profit:
                            best_profit = profit
                            best_plan = (tc, td, charge_dur, discharge_dur)

        power = np.zeros(96)
        if best_plan is not None:
            tc, td, charge_dur, discharge_dur = best_plan
            power[tc:tc+charge_dur] = -1000
            power[td:td+discharge_dur] = 1000
            total_profit += best_profit
            print(f"{date}: tc={tc}(+{charge_dur}), td={td}(+{discharge_dur}), profit={best_profit:.2f}")
        else:
            print(f"{date}: 无交易")

        for i, (t, p, pr) in enumerate(zip(times, power, prices)):
            results.append({'times': t, '实时价格': pr, 'power': p})

    df_result = pd.DataFrame(results)
    df_result.to_csv(save_path, index=False)

    n_days = len(df.groupby("date"))
    avg_profit = total_profit / n_days if n_days > 0 else 0
    print(f'v1动态策略 - 总天数:{n_days}, 总收益:{total_profit:.2f}, 日均收益:{avg_profit:.2f}')
    return avg_profit

if __name__ == '__main__':
    print("=" * 60)
    print("v1 多时段动态策略 - 可变时长充放电")
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
    avg_profit = generate_strategy_v1(output_price_path, output_power_path)

    print(f"\nv1完成! 提交文件: {output_power_path}")
    print(f"日均收益: {avg_profit:.2f}")