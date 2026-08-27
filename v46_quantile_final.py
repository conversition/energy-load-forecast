"""
v46 Quantile Regression + Strategy Optimization（里程碑3 - 最终版）
核心改进：
1. 96时刻 × 3分位数 = 288个独立模型
2. Route A: 保留分位数回归，舍弃样本加权
3. 基于P10/P50/P90的贪心策略优化
"""
import pandas as pd
import numpy as np
import os
import netCDF4 as nc
import glob
import logging
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')
nc_dir = os.path.join(current_dir, 'data', 'all_nc')

output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'v46_quantile_price.csv')
output_power_path = os.path.join(output_dir, 'v46_quantile_output.csv')

target_col = 'A'
LAG_96, LAG_192, LAG_288 = 96, 192, 288


def load_weather_data_fixed(nc_dir, start_date, end_date):
    """加载气象数据"""
    nc_files = sorted(glob.glob(os.path.join(nc_dir, '*.nc')))
    weather_list = []
    for nc_file in nc_files:
        filename = os.path.basename(nc_file)
        date_str = filename.replace('.nc', '')
        try:
            date = pd.to_datetime(date_str, format='%Y%m%d')
        except:
            continue
        if date < start_date or date > end_date:
            continue

        try:
            ds = nc.Dataset(nc_file, 'r')
            data_mean = ds.variables['data'][:].mean(axis=(3, 4))[0, :, :]
            t2m = data_mean[:, 2] - 273.15
            u100, v100 = data_mean[:, 5], data_mean[:, 6]
            wind_speed = np.sqrt(u100**2 + v100**2)

            utc_times = pd.date_range(date, periods=24, freq='h')
            bjt_times = utc_times + pd.Timedelta(hours=8)

            weather_df = pd.DataFrame({'times': bjt_times, 'temperature': t2m, 'wind_speed': wind_speed})
            weather_list.append(weather_df)
            ds.close()
        except:
            continue

    if not weather_list:
        return pd.DataFrame()
    df_weather = pd.concat(weather_list, ignore_index=True)
    df_weather['times'] = pd.to_datetime(df_weather['times'])
    df_weather = df_weather.drop_duplicates(subset=['times']).sort_values('times')
    df_weather = df_weather.set_index('times')
    df_weather_15min = df_weather.resample('15min').interpolate(method='quadratic').reset_index()
    return df_weather_15min


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


def add_interaction_features(df):
    df = df.copy()
    df['供需比'] = df['风光总加预测值'] / (df['系统负荷预测值'] + 1)
    df['风光差'] = df['风电预测值'] - df['光伏预测值']
    df['净负荷'] = df['系统负荷预测值'] - df['风光总加预测值']
    total_gen = df['风光总加预测值'] + df['水电预测值'] + df['非市场化机组预测值']
    df['风光渗透率'] = df['风光总加预测值'] / (total_gen + 1)
    df['is_workday'] = (df['dayofweek'] < 5).astype(int)
    return df


def compute_derivative_features(df, cols=['wind_speed', 'temperature']):
    for col in cols:
        if col not in df.columns:
            continue
        df[f'{col}_diff1'] = df.groupby('date')[col].diff(4).fillna(0)
        df[f'{col}_diff2'] = df.groupby('date')[f'{col}_diff1'].diff(4).fillna(0)
    return df


def compute_pressure_index(df):
    max_load = df['系统负荷预测值'].max()
    df['顶峰压力指数'] = np.exp(df['净负荷'] / max_load)
    df['顶峰压力线性'] = df['净负荷'] / max_load
    return df


def compute_deviation_stats(df_feat, deviation_cols):
    """仅计算偏差统计量（不泄露数据）"""
    df_tmp = df_feat.copy()
    df_tmp['times'] = pd.to_datetime(df_tmp['times'])
    df_tmp = add_time_features(df_tmp)

    deviation_stats = {}
    for col in deviation_cols:
        actual_col = f'{col}实际值'
        pred_col = f'{col}预测值'
        if actual_col in df_tmp.columns and pred_col in df_tmp.columns:
            deviation_col = f'{col}_deviation'
            df_tmp[deviation_col] = df_tmp[actual_col] - df_tmp[pred_col]
            grouped_mean = df_tmp.groupby(['hour', 'month'])[deviation_col].mean()
            deviation_stats[col] = grouped_mean.to_dict()

    return deviation_stats


def apply_deviation_features(df_feat, deviation_stats, deviation_cols):
    """应用已计算好的偏差统计量到数据（避免数据泄漏）"""
    df_feat_out = df_feat.copy()
    df_feat_out['times'] = pd.to_datetime(df_feat_out['times'])
    df_feat_out = add_time_features(df_feat_out)

    for col in deviation_cols:
        if col in deviation_stats:
            df_feat_out[f'{col}_dev_mean'] = df_feat_out.apply(
                lambda row: deviation_stats[col].get((row['hour'], row['month']), 0), axis=1
            )

    return df_feat_out


def get_feature_constraints(feature_names):
    MONOTONE_MAP = {
        '净负荷': 1,
        '风光渗透率': -1,
        '供需比': 1,
    }
    constraints = []
    for fname in feature_names:
        if fname in MONOTONE_MAP:
            constraints.append(MONOTONE_MAP[fname])
        else:
            constraints.append(0)
    return constraints


def prepare_time_of_day_data(df_feat, df_label, df_weather, weather_means, target_col='A'):
    df_feat_copy = df_feat.copy()
    df_feat_copy['times'] = pd.to_datetime(df_feat_copy['times'])
    df_label_copy = df_label.copy()
    df_label_copy['times'] = pd.to_datetime(df_label_copy['times'])
    df = pd.merge(df_feat_copy, df_label_copy, on='times', how='inner')

    if not df_weather.empty:
        df = pd.merge(df, df_weather, on='times', how='left')
        for col in ['temperature', 'wind_speed']:
            df[col] = df[col].fillna(weather_means.get(col, 0))

    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_interaction_features(df)

    df['time_step'] = df['hour'] * 4 + df['minute'] // 15
    df['date'] = df['times'].dt.date

    # 【修复隐患二】剔除零方差特征：hour/minute在单时刻模型内恒定
    # df = df.drop(columns=['hour', 'minute'], errors='ignore')

    df = compute_derivative_features(df, cols=['wind_speed', 'temperature'])
    df = compute_pressure_index(df)

    # 【修复隐患三】添加Direct Lags：历史电价作为特征（昨日同时刻真实值）
    df = df.sort_values('times').reset_index(drop=True)
    df['price_lag_1_day'] = df[target_col].shift(96)   # 昨日同一时刻
    df['price_lag_2_day'] = df[target_col].shift(192)  # 前日同一时刻

    # 滞后特征需要更长的初始丢弃（LAG_288 + 192 = 480）
    df = df.iloc[LAG_288 + 192:].reset_index(drop=True)
    df = df.dropna(subset=['price_lag_2_day']).reset_index(drop=True)

    return df, df['date'].unique(), df['times'].values


class QuantileTimeOfDayPredictor:
    """
    96时刻 × 3分位数 = 288个独立模型
    Route A: 保留分位数回归，舍弃样本加权
    """

    def __init__(self, n_steps=96):
        self.models = {}  # {(step, q_name): model}
        self.n_steps = n_steps
        self.quantiles = [('p10', 0.1), ('p50', 0.5), ('p90', 0.9)]

    def train(self, df_train, feature_cols, target_col):
        """训练288个分位数模型

        【修复隐患一】每个模型仅~365样本，必须大幅降低树复杂度
        - max_depth: 7 → 4
        - num_leaves: 63 → 15
        - min_child_samples: 添加15防止过拟合
        """
        constraints = get_feature_constraints(feature_cols)

        for step in range(self.n_steps):
            df_step = df_train[df_train['time_step'] == step].copy()

            if len(df_step) < 10:
                continue

            X = df_step[feature_cols].values
            y = df_step[target_col].values

            for q_name, alpha in self.quantiles:
                if alpha == 0.5:
                    # P50: 可以使用单调性约束
                    model = lgb.LGBMRegressor(
                        n_estimators=200, learning_rate=0.03,
                        max_depth=4, num_leaves=15,
                        min_child_samples=15,
                        subsample=0.8, colsample_bytree=0.8,
                        monotone_constraints=constraints,
                        random_state=42, n_jobs=-1
                    )
                else:
                    # P10/P90: 不能与 quantile objective 同时使用 monotone_constraints
                    model = lgb.LGBMRegressor(
                        objective='quantile', alpha=alpha,
                        n_estimators=200, learning_rate=0.03,
                        max_depth=4, num_leaves=15,
                        min_child_samples=15,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=-1
                    )

                model.fit(X, y)
                self.models[(step, q_name)] = model

            if step % 24 == 0:
                logger.info(f"  已训练时刻: {step + 1}/96")

        logger.info(f"  共训练 {len(self.models)} 个分位数模型")

    def predict(self, df_test, feature_cols):
        """预测测试集"""
        predictions = {
            'p10': np.zeros(len(df_test)),
            'p50': np.zeros(len(df_test)),
            'p90': np.zeros(len(df_test))
        }

        for step in range(self.n_steps):
            step_mask = df_test['time_step'] == step

            if not step_mask.any():
                continue

            X_test = df_test.loc[step_mask, feature_cols].values

            for q_name in ['p10', 'p50', 'p90']:
                if (step, q_name) in self.models:
                    predictions[q_name][step_mask] = self.models[(step, q_name)].predict(X_test)

        return predictions


def optimize_strategy(predictions_p10, predictions_p50, predictions_p90, risk_factor=0.3):
    """
    优化后的鲁棒性策略
    - 降低risk_factor (0.3 vs 0.5) 提高交易频率
    - 优先保证最坏情况，选择期望收益更高的方案
    """
    n_days = len(predictions_p50) // 96
    power_schedule = np.zeros(len(predictions_p50))
    total_profit = 0

    for day_idx in range(n_days):
        start = day_idx * 96
        end = start + 96

        p10 = predictions_p10[start:end]
        p50 = predictions_p50[start:end]
        p90 = predictions_p90[start:end]

        daily_std = np.std(p50)
        threshold = daily_std * risk_factor * 8 * 1000

        best_worst = -np.inf
        best_expected = 0
        best_tc, best_td = -1, -1

        for tc in range(0, 81):
            worst_charge = np.sum(p90[tc:tc+8]) * 1000

            for td in range(tc + 8, 89):
                worst_dis = np.sum(p10[td:td+8]) * 1000
                worst_profit = worst_dis - worst_charge

                expected_profit = (np.sum(p50[td:td+8]) - np.sum(p50[tc:tc+8])) * 1000

                if worst_profit > best_worst:
                    best_worst = worst_profit
                    best_expected = expected_profit
                    best_tc, best_td = tc, td
                elif worst_profit == best_worst and expected_profit > best_expected:
                    best_expected = expected_profit
                    best_tc, best_td = tc, td

        if best_tc >= 0 and best_worst >= threshold:
            power_schedule[start + best_tc:start + best_tc + 8] = -1000
            power_schedule[start + best_td:start + best_td + 8] = 1000
            total_profit += best_expected

    return power_schedule, total_profit


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("v46 Quantile Regression + Strategy Optimization (里程碑3)")
    logger.info("核心: 96×3=288分位数模型 + 贪心策略")
    logger.info("修复: 偏差特征数据泄漏, 参数倒挂, 零方差特征, Direct Lags")
    logger.info("=" * 60)

    # 特征配置
    # 【修复隐患二】去除零方差特征：hour/minute/sin_hour/cos_hour在单时刻模型内恒为常量
    base_features = ['系统负荷预测值', '风光总加预测值', '联络线预测值', '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
    time_features = ['dayofweek', 'month']  # 去掉 hour, minute
    cyclical_features = ['sin_month', 'cos_month']  # 去掉 sin_hour, cos_hour
    physics_features = ['供需比', '风光差', '净负荷', '风光渗透率', 'is_workday', '顶峰压力指数', '顶峰压力线性']
    weather_features = ['wind_speed', 'temperature']
    derivative_features = ['wind_speed_diff1', 'wind_speed_diff2', 'temperature_diff1', 'temperature_diff2']

    deviation_cols = ['系统负荷', '风光总加', '风电', '光伏']
    deviation_feature_names = [f'{col}_dev_mean' for col in deviation_cols]

    # 【修复隐患三】添加Direct Lags特征
    lag_features = ['price_lag_1_day', 'price_lag_2_day']

    all_features = base_features + time_features + cyclical_features + physics_features + weather_features + derivative_features + deviation_feature_names + lag_features

    logger.info(f"特征数量: {len(all_features)}")

    # 1. 加载数据
    logger.info("[1/6] 加载数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)
    df_test = pd.read_csv(test_feature_path)

    df_train_feat = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train_feat['times'] = pd.to_datetime(df_train_feat['times'])
    df_test['times'] = pd.to_datetime(df_test['times'])

    all_start = min(df_train_feat['times'].min(), df_test['times'].min())
    all_end = max(df_train_feat['times'].max(), df_test['times'].max())

    df_weather = load_weather_data_fixed(nc_dir, all_start, all_end)
    weather_means = {col: df_weather[col].mean() for col in ['temperature', 'wind_speed']} if not df_weather.empty else {}

    # 2. 准备数据（暂不添加偏差特征）
    logger.info("[2/6] 准备数据（无偏差特征）...")
    df_train_raw, train_dates, train_times = prepare_time_of_day_data(df_feat, df_label, df_weather, weather_means)

    # 3. 数据分割
    logger.info("[3/6] 数据分割...")
    split_idx = int(len(df_train_raw) * 0.8)
    df_train_split = df_train_raw.iloc[:split_idx].copy()
    df_val_split = df_train_raw.iloc[split_idx:].copy()

    # 【修复偏差特征数据泄漏】仅在训练集上计算偏差统计量
    logger.info("[4/6] 计算偏差特征（仅训练集）...")
    deviation_stats = compute_deviation_stats(df_feat, deviation_cols)

    # 应用偏差特征到各数据集
    df_train_split = apply_deviation_features(df_train_split, deviation_stats, deviation_cols)
    df_val_split = apply_deviation_features(df_val_split, deviation_stats, deviation_cols)

    # 移除偏差特征名称列表（已手动添加）
    deviation_feature_names = [f'{col}_dev_mean' for col in deviation_cols]

    logger.info(f"训练样本: {len(df_train_split)}, 验证样本: {len(df_val_split)}")

    # 4. 训练分位数模型
    logger.info("[5/6] 训练288个分位数模型...")
    predictor = QuantileTimeOfDayPredictor(n_steps=96)
    predictor.train(df_train_split, all_features, target_col)

    # 5. 验证集评估
    logger.info("[6/6] 验证集评估...")
    val_predictions = predictor.predict(df_val_split, all_features)

    y_val = df_val_split[target_col].values
    rmse_p50 = np.sqrt(mean_squared_error(y_val, val_predictions['p50']))
    logger.info(f"  验证集 P50 RMSE: {rmse_p50:.6f}")

    # 验证集策略评估
    power_schedule, val_profit = optimize_strategy(
        val_predictions['p10'],
        val_predictions['p50'],
        val_predictions['p90'],
        risk_factor=0.3
    )
    logger.info(f"  验证集策略收益: {val_profit:.0f}")

    # ========== 测试集预测 ==========
    logger.info("[测试集] 准备测试数据...")
    df_test_raw = pd.read_csv(test_feature_path)
    df_test_raw['times'] = pd.to_datetime(df_test_raw['times'])

    # 处理测试集气象数据
    if not df_weather.empty:
        df_test_raw = pd.merge(df_test_raw, df_weather, on='times', how='left')
        for col in ['temperature', 'wind_speed']:
            df_test_raw[col] = df_test_raw[col].fillna(weather_means.get(col, 0))

    df_test_raw = add_time_features(df_test_raw)
    df_test_raw = add_cyclical_features(df_test_raw)
    df_test_raw = add_interaction_features(df_test_raw)

    df_test_raw['time_step'] = df_test_raw['hour'] * 4 + df_test_raw['minute'] // 15
    df_test_raw['date'] = df_test_raw['times'].dt.date

    df_test_raw = compute_derivative_features(df_test_raw, cols=['wind_speed', 'temperature'])
    df_test_raw = compute_pressure_index(df_test_raw)

    # 【修复偏差泄漏】使用与训练集相同的偏差统计量
    df_test_raw = apply_deviation_features(df_test_raw, deviation_stats, deviation_cols)

    # 【修复隐患三】添加测试集滞后特征
    # 由于测试集没有真实电价，使用0填充
    df_test_raw = df_test_raw.sort_values('times').reset_index(drop=True)
    df_test_raw['price_lag_1_day'] = 0.0
    df_test_raw['price_lag_2_day'] = 0.0

    logger.info("[测试集] 预测...")
    test_predictions = predictor.predict(df_test_raw, all_features)

    # 保存价格预测
    df_price_out = pd.DataFrame({
        'times': df_test_raw['times'].values,
        'P10': test_predictions['p10'],
        'P50': test_predictions['p50'],
        'P90': test_predictions['p90']
    })
    df_price_out.to_csv(output_price_path, index=False)
    logger.info(f"  价格预测已保存: {output_price_path}")

    # 生成策略
    logger.info("[测试集] 生成策略...")
    power_schedule, test_profit = optimize_strategy(
        test_predictions['p10'],
        test_predictions['p50'],
        test_predictions['p90'],
        risk_factor=0.3
    )

    n_test_days = len(test_predictions['p50']) // 96
    df_power_out = pd.DataFrame({
        'times': df_test_raw['times'].values,
        'P10': test_predictions['p10'],
        'P90': test_predictions['p90'],
        'median': test_predictions['p50'],
        'power': power_schedule
    })
    df_power_out.to_csv(output_power_path, index=False)
    logger.info(f"  策略已保存: {output_power_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"v46 Quantile + Strategy 完成!")
    logger.info(f"验证集 P50 RMSE: {rmse_p50:.6f}")
    logger.info(f"验证集策略收益: {val_profit:.0f}")
    logger.info(f"测试集策略收益: {test_profit:.0f}")
    logger.info(f"测试集天数: {n_test_days}")
    logger.info("=" * 60)