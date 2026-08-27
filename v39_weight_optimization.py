"""
v39 融合权重优化版 - 在v37基础上优化GBDT+LightGBM融合权重
核心优化:
1. 基于验证集表现动态调整融合权重
2. 尝试不同的权重组合找到最优
3. 结合v37的成功经验(wind+temperature)
"""
import pandas as pd
import numpy as np
import os
import netCDF4 as nc
import glob
import logging
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
output_price_path = os.path.join(output_dir, 'v39_weight_opt_price.csv')
output_power_path = os.path.join(output_dir, 'v39_weight_opt_output.csv')

target_col = 'A'
LAG_96, LAG_192, LAG_288 = 96, 192, 288


def load_weather_data_fixed(nc_dir, start_date, end_date):
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
            ghi = data_mean[:, 0]
            t2m = data_mean[:, 2] - 273.15
            u100, v100 = data_mean[:, 5], data_mean[:, 6]
            wind_speed = np.sqrt(u100**2 + v100**2)

            utc_times = pd.date_range(date, periods=24, freq='h')
            bjt_times = utc_times + pd.Timedelta(hours=8)

            weather_df = pd.DataFrame({'times': bjt_times, 'temperature': t2m, 'ghi': ghi, 'wind_speed': wind_speed})
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


def add_business_features(df):
    df = df.copy()
    df['净负荷'] = df['系统负荷预测值'] - df['风光总加预测值']
    total_gen = df['风光总加预测值'] + df['水电预测值'] + df['非市场化机组预测值']
    df['风光渗透率'] = df['风光总加预测值'] / (total_gen + 1)
    return df


def prepare_train_data(df_feat, df_label, df_weather, weather_means):
    df = pd.merge(df_feat, df_label, on='times', how='inner')
    df['times'] = pd.to_datetime(df['times'])
    if not df_weather.empty:
        df = pd.merge(df, df_weather, on='times', how='left')
        for col in ['temperature', 'ghi', 'wind_speed']:
            df[col] = df[col].fillna(weather_means.get(col, 0))
    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_business_features(df)
    df['price_lag_96'] = df[target_col].shift(LAG_96)
    df['price_lag_192'] = df[target_col].shift(LAG_192)
    rolling = df['price_lag_96'].rolling(window=LAG_96, min_periods=1)
    df['rolling_mean'] = rolling.mean()
    df['rolling_std'] = rolling.std()
    return df.iloc[LAG_192:].reset_index(drop=True)


def simulate_true_validation_profit(prices_pred, prices_true, noise_margin=0.0, threshold=0):
    n_days = len(prices_pred) // 96
    total_true_profit = 0
    for day_idx in range(n_days):
        start = day_idx * 96
        end = start + 96
        day_pred = prices_pred[start:end]
        day_true = prices_true[start:end]
        if len(day_pred) != 96:
            continue

        best_worst, best_tc, best_td = -float('inf'), -1, -1
        for tc in range(0, 81):
            worst_charge = np.sum(day_pred[tc:tc+8]) * 1000 * (1 + noise_margin)
            for td in range(tc + 8, 89):
                worst_dis = np.sum(day_pred[td:td+8]) * 1000 * (1 - noise_margin)
                if worst_dis - worst_charge > best_worst:
                    best_worst = worst_dis - worst_charge
                    best_tc, best_td = tc, td

        if best_tc >= 0 and best_worst >= threshold:
            total_true_profit += (np.sum(day_true[best_td:best_td+8]) - np.sum(day_true[best_tc:best_tc+8])) * 1000
    return total_true_profit


def generate_strategy(price_csv, save_path, noise_margin=0.0, threshold=0):
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])
    df['date'] = df['times'].dt.date
    results, total_profit, days_operated = [], 0, 0

    for date, group in df.groupby('date'):
        prices = group['A'].values
        times = group['times'].values
        if len(prices) != 96:
            continue

        best_worst, best_tc, best_td = -float('inf'), -1, -1
        for tc in range(0, 81):
            worst_charge = np.sum(prices[tc:tc+8]) * 1000 * (1 + noise_margin)
            for td in range(tc + 8, 89):
                worst_dis = np.sum(prices[td:td+8]) * 1000 * (1 - noise_margin)
                if worst_dis - worst_charge > best_worst:
                    best_worst = worst_dis - worst_charge
                    best_tc, best_td = tc, td

        power = np.zeros(96)
        if best_tc >= 0 and best_worst >= threshold:
            power[best_tc:best_tc+8], power[best_td:best_td+8] = -1000, 1000
            total_profit += (np.sum(prices[best_td:best_td+8]) - np.sum(prices[best_tc:best_tc+8])) * 1000
            days_operated += 1

        results.extend([{'times': t, '实时价格': p, 'power': pw} for t, pw, p in zip(times, power, prices)])

    pd.DataFrame(results).to_csv(save_path, index=False)
    logger.info(f"  执行策略: noise={noise_margin}, threshold={threshold}, 操作 {days_operated} 天")
    return total_profit


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("v39 融合权重优化版 - 动态调整GBDT+LightGBM权重")
    logger.info("=" * 60)

    base_features = ['系统负荷预测值', '风光总加预测值', '联络线预测值', '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
    time_features = ['hour', 'minute', 'dayofweek']
    cyclical_features = ['sin_month', 'cos_month', 'sin_hour', 'cos_hour']
    business_features = ['净负荷', '风光渗透率']
    lag_features = ['price_lag_96', 'price_lag_192']
    stat_features = ['rolling_mean', 'rolling_std']

    final_weather_features = ['wind_speed', 'temperature']
    all_features = base_features + time_features + cyclical_features + business_features + lag_features + stat_features + final_weather_features

    logger.info(f"特征配置 ({len(all_features)}个): {final_weather_features}")

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
    weather_means = {col: df_weather[col].mean() for col in ['temperature', 'ghi', 'wind_speed']} if not df_weather.empty else {}

    df_train = prepare_train_data(df_feat, df_label, df_weather, weather_means)
    y = df_train[target_col].values
    split_idx = int(len(y) * 0.8)
    X = df_train[all_features].values
    X_train, X_val, y_train, y_val = X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    logger.info(f"训练样本: {len(df_train)}, 训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # 2. 训练双模型
    logger.info("[2/6] 训练双模型 (GBDT + LightGBM)...")

    model_gbdt = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42)
    model_gbdt.fit(X_train, y_train)

    model_lgb = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=7, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model_lgb.fit(X_train, y_train)

    # 3. 获取验证集预测
    logger.info("[3/6] 验证集预测...")
    pred_gbdt_val = model_gbdt.predict(X_val)
    pred_lgb_val = model_lgb.predict(X_val)

    # 4. 寻优融合权重和策略参数
    logger.info("[4/6] 寻优融合权重和策略参数...")

    best_overall_profit = -float('inf')
    best_config = {}

    weight_configs = [0.3, 0.4, 0.5, 0.6, 0.7]
    noise_configs = [0.0, 0.05, 0.10]
    threshold_configs = [0, 1000]

    for w_gbdt in weight_configs:
        w_lgb = 1.0 - w_gbdt
        pred_ensemble = pred_gbdt_val * w_gbdt + pred_lgb_val * w_lgb

        for noise in noise_configs:
            for threshold in threshold_configs:
                profit = simulate_true_validation_profit(pred_ensemble, y_val, noise, threshold)
                if profit > best_overall_profit:
                    best_overall_profit = profit
                    best_config = {
                        'w_gbdt': w_gbdt, 'w_lgb': w_lgb,
                        'noise': noise, 'threshold': threshold,
                        'profit': profit
                    }

    logger.info(f"  最优配置: GBDT权重={best_config['w_gbdt']}, LightGBM权重={best_config['w_lgb']}")
    logger.info(f"  策略参数: noise={best_config['noise']}, threshold={best_config['threshold']}")
    logger.info(f"  真实验证收益: {best_config['profit']:.0f}")

    w_gbdt = best_config['w_gbdt']
    w_lgb = best_config['w_lgb']
    best_noise = best_config['noise']
    best_threshold = best_config['threshold']

    # 5. 测试集递推预测
    logger.info("[5/6] 测试集双模型递推预测...")

    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])

    if not df_weather.empty:
        df_test = pd.merge(df_test, df_weather, on='times', how='left')
        for col in ['temperature', 'ghi', 'wind_speed']:
            df_test[col] = df_test[col].fillna(weather_means.get(col, 0))

    df_test = add_time_features(df_test)
    df_test = add_cyclical_features(df_test)
    df_test = add_business_features(df_test)

    base_test_features = base_features + time_features + cyclical_features + business_features + final_weather_features

    predicted_prices = np.zeros(len(df_test))
    all_predicted = np.concatenate([df_train[target_col].values[-LAG_192:].copy()])
    n_days = len(df_test) // 96

    logger.info(f"测试集天数: {n_days}")

    for day_idx in range(n_days):
        start_idx = day_idx * 96
        end_idx = start_idx + 96

        day_X = df_test[base_test_features].iloc[start_idx:end_idx].values.copy()

        history = pd.Series(all_predicted[-LAG_288:])
        lag_96 = history.shift(LAG_96).values[-LAG_96:]
        lag_192 = history.shift(LAG_192).values[-LAG_96:]
        roll_mean = history.shift(LAG_96).rolling(LAG_96, min_periods=1).mean().values[-LAG_96:]
        roll_std = history.shift(LAG_96).rolling(LAG_96, min_periods=1).std().values[-LAG_96:]

        mean_val = np.nanmean(all_predicted)
        std_val = np.nanstd(all_predicted)
        lag_96 = np.nan_to_num(lag_96, nan=mean_val)
        lag_192 = np.nan_to_num(lag_192, nan=mean_val)
        roll_mean = np.nan_to_num(roll_mean, nan=mean_val)
        roll_std = np.nan_to_num(roll_std, nan=std_val)

        day_X_with_lag = np.column_stack([day_X, lag_96, lag_192, roll_mean, roll_std])

        day_pred_gbdt = model_gbdt.predict(day_X_with_lag)
        day_pred_lgb = model_lgb.predict(day_X_with_lag)
        day_pred_ensemble = day_pred_gbdt * w_gbdt + day_pred_lgb * w_lgb

        predicted_prices[start_idx:end_idx] = day_pred_ensemble
        all_predicted = np.concatenate([all_predicted, day_pred_ensemble])

        if day_idx < 3 or day_idx == n_days - 1:
            logger.info(f"  Day {day_idx+1}: mean={day_pred_ensemble.mean():.4f}")

    df_out = pd.DataFrame({'times': df_test['times'], target_col: predicted_prices})
    df_out.to_csv(output_price_path, index=False)

    # 6. 生成最终策略
    logger.info("[6/6] 生成最终充放电策略...")
    total_profit = generate_strategy(output_price_path, output_power_path, best_noise, best_threshold)

    logger.info("\n" + "=" * 60)
    logger.info(f"v39 融合权重优化版完成!")
    logger.info(f"最优权重: GBDT={w_gbdt}, LightGBM={w_lgb}")
    logger.info(f"最优策略: noise={best_noise}, threshold={best_threshold}")
    logger.info(f"测试集总收益: {total_profit:.2f}, 日均: {total_profit/n_days:.2f}")
    logger.info("=" * 60)