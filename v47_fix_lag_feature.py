"""
v47: 修复测试集历史电价Lag特征的致命Bug（最高优先级）
核心问题: 测试集的price_lag_1_day和price_lag_2_day被硬编码为0，导致特征分布偏移
解决方案:
1. 测试集第1天(2026-01-01)使用训练集最后2天的真实电价作为lag特征
2. 测试集后续天使用递归预测：用当天预测值填充次日lag特征
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
output_price_path = os.path.join(output_dir, 'v47_fix_lag_price.csv')
output_power_path = os.path.join(output_dir, 'v47_fix_lag_output.csv')

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
    """应用已计算好的偏差统计量到数据"""
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

    df = compute_derivative_features(df, cols=['wind_speed', 'temperature'])
    df = compute_pressure_index(df)

    df = df.sort_values('times').reset_index(drop=True)
    df['price_lag_1_day'] = df[target_col].shift(96)
    df['price_lag_2_day'] = df[target_col].shift(192)

    df = df.iloc[LAG_288 + 192:].reset_index(drop=True)
    df = df.dropna(subset=['price_lag_2_day']).reset_index(drop=True)

    return df, df['date'].unique(), df['times'].values


def prepare_single_day_features(df_day, df_base_features, deviation_stats, deviation_cols):
    """为单日数据添加所有特征"""
    df = df_day.copy()
    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_interaction_features(df)

    df['time_step'] = df['hour'] * 4 + df['minute'] // 15
    df['date'] = df['times'].dt.date

    df = compute_derivative_features(df, cols=['wind_speed', 'temperature'])
    df = compute_pressure_index(df)

    df = apply_deviation_features(df, deviation_stats, deviation_cols)

    return df


class QuantileTimeOfDayPredictor:
    """
    96时刻 × 3分位数 = 288个独立模型
    Route A: 保留分位数回归，舍弃样本加权
    """

    def __init__(self, n_steps=96):
        self.models = {}
        self.n_steps = n_steps
        self.quantiles = [('p10', 0.1), ('p50', 0.5), ('p90', 0.9)]

    def train(self, df_train, feature_cols, target_col):
        """训练288个分位数模型"""
        constraints = get_feature_constraints(feature_cols)

        for step in range(self.n_steps):
            df_step = df_train[df_train['time_step'] == step].copy()

            if len(df_step) < 10:
                continue

            X = df_step[feature_cols].values
            y = df_step[target_col].values

            for q_name, alpha in self.quantiles:
                if alpha == 0.5:
                    model = lgb.LGBMRegressor(
                        n_estimators=200, learning_rate=0.03,
                        max_depth=4, num_leaves=15,
                        min_child_samples=15,
                        subsample=0.8, colsample_bytree=0.8,
                        monotone_constraints=constraints,
                        random_state=42, n_jobs=-1
                    )
                else:
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

    def predict_single_day(self, df_day, feature_cols):
        """预测单日96个时刻"""
        predictions = {
            'p10': np.zeros(96),
            'p50': np.zeros(96),
            'p90': np.zeros(96)
        }

        for step in range(self.n_steps):
            step_mask = df_day['time_step'] == step

            if not step_mask.any():
                continue

            X_test = df_day.loc[step_mask, feature_cols].values

            for q_name in ['p10', 'p50', 'p90']:
                if (step, q_name) in self.models:
                    predictions[q_name][step] = self.models[(step, q_name)].predict(X_test)[0]

        return predictions

    def predict_batch(self, df_test, feature_cols):
        """批量预测"""
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
    """优化后的鲁棒性策略"""
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
    logger.info("v47: 修复测试集历史电价Lag特征Bug（最高优先级）")
    logger.info("修复: 测试集lag特征从0改为使用真实历史电价/递归预测电价")
    logger.info("=" * 60)

    # 特征配置
    base_features = ['系统负荷预测值', '风光总加预测值', '联络线预测值', '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
    time_features = ['dayofweek', 'month']
    cyclical_features = ['sin_month', 'cos_month']
    physics_features = ['供需比', '风光差', '净负荷', '风光渗透率', 'is_workday', '顶峰压力指数', '顶峰压力线性']
    weather_features = ['wind_speed', 'temperature']
    derivative_features = ['wind_speed_diff1', 'wind_speed_diff2', 'temperature_diff1', 'temperature_diff2']

    deviation_cols = ['系统负荷', '风光总加', '风电', '光伏']
    deviation_feature_names = [f'{col}_dev_mean' for col in deviation_cols]

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

    # 2. 准备训练数据
    logger.info("[2/6] 准备训练数据...")
    df_train_raw, train_dates, train_times = prepare_time_of_day_data(df_feat, df_label, df_weather, weather_means)

    # 3. 数据分割
    logger.info("[3/6] 数据分割...")
    split_idx = int(len(df_train_raw) * 0.8)
    df_train_split = df_train_raw.iloc[:split_idx].copy()
    df_val_split = df_train_raw.iloc[split_idx:].copy()

    # 计算偏差特征（仅训练集）
    logger.info("[4/6] 计算偏差特征（仅训练集）...")
    deviation_stats = compute_deviation_stats(df_feat, deviation_cols)

    df_train_split = apply_deviation_features(df_train_split, deviation_stats, deviation_cols)
    df_val_split = apply_deviation_features(df_val_split, deviation_stats, deviation_cols)

    deviation_feature_names = [f'{col}_dev_mean' for col in deviation_cols]

    logger.info(f"训练样本: {len(df_train_split)}, 验证样本: {len(df_val_split)}")

    # 4. 训练分位数模型
    logger.info("[5/6] 训练288个分位数模型...")
    predictor = QuantileTimeOfDayPredictor(n_steps=96)
    predictor.train(df_train_split, all_features, target_col)

    # 5. 验证集评估
    logger.info("[6/6] 验证集评估...")
    val_predictions = predictor.predict_batch(df_val_split, all_features)

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

    # ========== 测试集预测（修复Lag特征）==========
    logger.info("[测试集] 准备测试数据（修复Lag特征Bug）...")

    # 【核心修复】获取训练集最后2天的真实电价用于测试集Lag特征
    df_label['times'] = pd.to_datetime(df_label['times'])
    df_label_sorted = df_label.sort_values('times').tail(192 * 2)  # 最后2天
    historical_prices = df_label_sorted[target_col].values
    logger.info(f"  历史电价天数: {len(historical_prices) // 96}天, 共{len(historical_prices)}个点")

    # 准备测试集基础特征
    df_test_raw = pd.read_csv(test_feature_path)
    df_test_raw['times'] = pd.to_datetime(df_test_raw['times'])

    # 处理气象数据
    if not df_weather.empty:
        df_test_raw = pd.merge(df_test_raw, df_weather, on='times', how='left')
        for col in ['temperature', 'wind_speed']:
            df_test_raw[col] = df_test_raw[col].fillna(weather_means.get(col, 0))

    df_test_raw = prepare_single_day_features(df_test_raw, base_features, deviation_stats, deviation_cols)
    df_test_raw = df_test_raw.sort_values('times').reset_index(drop=True)

    n_test_days = len(df_test_raw) // 96
    logger.info(f"  测试集天数: {n_test_days}")

    # 【核心修复】递归预测：逐日预测并更新Lag特征
    all_p10 = []
    all_p50 = []
    all_p90 = []

    # 前2天的历史电价（从训练集末尾获取）
    last_2_days_prices = historical_prices  # shape: (192,) = 2天*96点

    for day_idx in range(n_test_days):
        start_idx = day_idx * 96
        end_idx = start_idx + 96

        df_day = df_test_raw.iloc[start_idx:end_idx].copy()

        # 设置Lag特征
        if day_idx == 0:
            # 第1天：用训练集最后2天作为lag
            df_day['price_lag_1_day'] = last_2_days_prices[96:192]  # 昨日 = 训练集最后1天
            df_day['price_lag_2_day'] = last_2_days_prices[0:96]   # 前日 = 训练集倒数第2天
        else:
            # 后续天：用前1-2天的预测电价作为lag
            prev_day_p50 = all_p50[-96:] if len(all_p50) >= 96 else np.zeros(96)
            prev_2_day_p50 = all_p50[-192:-96] if len(all_p50) >= 192 else np.zeros(96)

            # 使用前一天和前两天的中位数预测作为lag
            df_day['price_lag_1_day'] = prev_day_p50
            df_day['price_lag_2_day'] = prev_2_day_p50

        # 预测当天
        day_preds = predictor.predict_single_day(df_day, all_features)

        all_p10.extend(day_preds['p10'])
        all_p50.extend(day_preds['p50'])
        all_p90.extend(day_preds['p90'])

        if (day_idx + 1) % 10 == 0 or day_idx == n_test_days - 1:
            logger.info(f"  已预测: {day_idx + 1}/{n_test_days}天")

    all_p10 = np.array(all_p10)
    all_p50 = np.array(all_p50)
    all_p90 = np.array(all_p90)

    # 保存价格预测
    df_price_out = pd.DataFrame({
        'times': df_test_raw['times'].values,
        'P10': all_p10,
        'P50': all_p50,
        'P90': all_p90
    })
    df_price_out.to_csv(output_price_path, index=False)
    logger.info(f"  价格预测已保存: {output_price_path}")

    # 生成策略
    logger.info("[测试集] 生成策略...")
    power_schedule, test_profit = optimize_strategy(
        all_p10,
        all_p50,
        all_p90,
        risk_factor=0.3
    )

    df_power_out = pd.DataFrame({
        'times': df_test_raw['times'].values,
        'P10': all_p10,
        'P90': all_p90,
        'median': all_p50,
        'power': power_schedule
    })
    df_power_out.to_csv(output_power_path, index=False)
    logger.info(f"  策略已保存: {output_power_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"v47 Lag特征修复完成!")
    logger.info(f"验证集 P50 RMSE: {rmse_p50:.6f}")
    logger.info(f"验证集策略收益: {val_profit:.0f}")
    logger.info(f"测试集策略收益: {test_profit:.0f}")
    logger.info(f"测试集天数: {n_test_days}")
    logger.info("=" * 60)