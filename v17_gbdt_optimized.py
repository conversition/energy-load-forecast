"""
v17 GBDT优化版 - 修复v16的核心问题
基于分析报告的四大问题修复:
1. 恢复被删除的核心特征: 风电预测值、联络线预测值、dayofweek、minute
2. 修正特征工程: 月份周期编码 + 净负荷
3. 换回GBDT模型 (ExtraTrees平滑了极端价格)
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
output_price_path = os.path.join(output_dir, 'v17_gbdt_price.csv')
output_power_path = os.path.join(output_dir, 'v17_gbdt_output.csv')

target_col = 'A'


def add_time_features(df):
    """添加时间特征 - 恢复v0基线的完整时间特征"""
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


def add_cyclical_features(df):
    """添加周期特征 - 修复季节性特征工程问题

    问题: month是1-12的周期变量，直接乘以负荷会在物理意义上让12月的权重是1月的12倍
    解决: 使用sin/cos周期编码，让12月和1月在特征空间中相邻
    """
    df = df.copy()
    # 月份周期编码
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    # 小时周期编码 (电力市场24小时循环)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


def add_business_features(df):
    """添加业务特征 - 修复净负荷计算

    问题: 负荷率=负荷/(风光+1) 在夜间风光接近0时会产生异常尖峰
    解决: 使用净负荷=负荷-风光，更符合电力调度业务逻辑，且数值稳定
    """
    df = df.copy()
    # 净负荷 - 核心业务特征，比负荷率稳定得多
    df['净负荷'] = df['系统负荷预测值'] - df['风光总加预测值']
    # 风光渗透率
    total_gen = df['风光总加预测值'] + df['水电预测值'] + df['非市场化机组预测值']
    df['风光渗透率'] = df['风光总加预测值'] / (total_gen + 1)
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
    print("v17 GBDT优化版 - 修复v16核心问题")
    print("=" * 60)

    # ==================== 数据准备 ====================
    print("\n[1/4] 加载数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)

    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    df_train = add_time_features(df_train)
    df_train = add_cyclical_features(df_train)
    df_train = add_business_features(df_train)

    # 核心特征 - 恢复v0基线完整特征 + 修复后的周期特征
    # v0基线: 系统负荷预测值, 风光总加预测值, 联络线预测值, 风电预测值, 光伏预测值, 水电预测值, 非市场化机组预测值
    base_features = [
        '系统负荷预测值', '风光总加预测值', '联络线预测值',
        '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值'
    ]
    time_features = ['hour', 'minute', 'dayofweek']
    cyclical_features = ['sin_month', 'cos_month', 'sin_hour', 'cos_hour']
    business_features = ['净负荷', '风光渗透率']

    all_features = base_features + time_features + cyclical_features + business_features
    print(f"  特征数量: {len(all_features)}")
    print(f"  特征列表: {all_features}")

    X = df_train[all_features].values
    y = df_train[target_col].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # ==================== 模型训练 ====================
    print("\n[2/4] 训练GBDT模型...")
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

    # ==================== 测试推理 ====================
    print("\n[3/4] 测试集推理...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_time_features(df_test)
    df_test = add_cyclical_features(df_test)
    df_test = add_business_features(df_test)

    X_test = df_test[all_features].values
    y_test_pred = model.predict(X_test)

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)

    # ==================== 生成策略 ====================
    print("\n[4/4] 生成充放电策略...")
    total_profit = generate_strategy(output_price_path, output_power_path)
    n_days = len(df_out) // 96
    print(f"\n  总收益: {total_profit:.2f}, 日均收益: {total_profit/n_days:.2f}")

    # 特征重要性
    print("\n" + "=" * 60)
    print("Feature Importance (v17 GBDT优化版)")
    print("=" * 60)
    for feat, imp in sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1]):
        bar = "=" * int(imp * 100)
        print(f"  {feat:<20s}: {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("v17 GBDT优化版完成!")
    print("=" * 60)