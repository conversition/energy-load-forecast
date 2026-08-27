"""
v38 分位数回归版 - 智能不确定性估计
核心优化:
1. 分位数回归: P10/P50/P90 模型预测最坏/最可能/最优情况
2. 动态波动率阈值: 根据每日价格波动率调整安全垫
3. 用机器学习的分位数代替人工设定的noise_margin
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
output_price_path = os.path.join(output_dir, 'v38_quantile_price.csv')
output_power_path = os.path.join(output_dir, 'v38_quantile_output.csv')

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


def generate_quantile_strategy(price_p10, price_p90, price_median, save_path, risk_factor=0.5):
    """基于分位数回归的智能策略"""
    df = pd.DataFrame({'p10': price_p10, 'p90': price_p90, 'median': price_median})
    df['date'] = [i // 96 for i in range(len(df))]
    results, total_profit, days_operated = [], 0, 0

    for day_idx in range(len(df) // 96):
        start = day_idx * 96
        end = start + 96
        if end > len(df):
            break

        p10_day = df['p10'].values[start:end]
        p90_day = df['p90'].values[start:end]
        median_day = df['median'].values[start:end]

        daily_std = np.std(median_day)
        dynamic_threshold = daily_std * risk_factor * 8 * 1000

        best_worst, best_tc, best_td = -float('inf'), -1, -1
        for tc in range(0, 81):
            worst_charge = np.sum(p90_day[tc:tc+8]) * 1000
            for td in range(tc + 8, 89):
                worst_dis = np.sum(p10_day[td:td+8]) * 1000
                worst_profit = worst_dis - worst_charge
                if worst_profit > best_worst:
                    best_worst = worst_profit
                    best_tc, best_td = tc, td

        power = np.zeros(96)
        if best_tc >= 0 and best_worst >= dynamic_threshold:
            power[best_tc:best_tc+8] = -1000
            power[best_td:best_td+8] = 1000
            total_profit += (np.sum(median_day[best_td:best_td+8]) - np.sum(median_day[best_tc:best_tc+8])) * 1000
            days_operated += 1

        results.extend([{'times': t, 'p10': p10, 'p90': p90, 'median': m, 'power': pw}
                        for t, p10, p90, m, pw in zip(range(start, end), p10_day, p90_day, median_day, power)])

    pd.DataFrame(results).to_csv(save_path, index=False)
    logger.info(f"  执行策略: risk_factor={risk_factor}, 操作 {days_operated} 天, 动态阈值")
    return total_profit


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("v38 分位数回归版 - P10/P50/P90 + 动态波动率阈值")
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

    # 2. 训练分位数模型组
    logger.info("[2/6] 训练分位数模型组 (P10/P50/P90)...")

    logger.info("  训练 P50 (中位数/最可能预测)...")
    model_p50 = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=7, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model_p50.fit(X_train, y_train)

    logger.info("  训练 P10 (放电下限/最悲观低价)...")
    model_p10 = lgb.LGBMRegressor(
        objective='quantile', alpha=0.1,
        n_estimators=300, learning_rate=0.03, max_depth=7, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model_p10.fit(X_train, y_train)

    logger.info("  训练 P90 (充电上限/最悲观高价)...")
    model_p90 = lgb.LGBMRegressor(
        objective='quantile', alpha=0.9,
        n_estimators=300, learning_rate=0.03, max_depth=7, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model_p90.fit(X_train, y_train)

    # 3. 验证集评估
    logger.info("[3/6] 验证集评估...")
    pred_p10_val = model_p10.predict(X_val)
    pred_p90_val = model_p90.predict(X_val)
    pred_p50_val = model_p50.predict(X_val)

    rmse_p50 = np.sqrt(mean_squared_error(y_val, pred_p50_val))
    logger.info(f"  P50 RMSE: {rmse_p50:.6f}")
    logger.info(f"  P10-P50 gap (avg): {np.mean(pred_p10_val - pred_p50_val):.4f}")
    logger.info(f"  P90-P50 gap (avg): {np.mean(pred_p90_val - pred_p50_val):.4f}")

    # 4. 寻优risk_factor参数
    logger.info("[4/6] 寻优risk_factor参数...")
    best_profit, best_rf = -float('inf'), 0.5
    for rf in [0.3, 0.5, 0.7, 1.0]:
        val_df = pd.DataFrame({'p10': pred_p10_val, 'p90': pred_p90_val, 'median': pred_p50_val})
        val_df['date'] = [i // 96 for i in range(len(val_df))]

        profit = 0
        for day_idx in range(len(val_df) // 96):
            start = day_idx * 96
            end = start + 96
            if end > len(val_df):
                break

            p10_day = val_df['p10'].values[start:end]
            p90_day = val_df['p90'].values[start:end]
            median_day = val_df['median'].values[start:end]
            y_day = y_val[start:end]

            daily_std = np.std(median_day)
            dynamic_threshold = daily_std * rf * 8 * 1000

            best_worst, best_tc, best_td = -float('inf'), -1, -1
            for tc in range(0, 81):
                worst_charge = np.sum(p90_day[tc:tc+8]) * 1000
                for td in range(tc + 8, 89):
                    worst_dis = np.sum(p10_day[td:td+8]) * 1000
                    if worst_dis - worst_charge > best_worst:
                        best_worst = worst_dis - worst_charge
                        best_tc, best_td = tc, td

            if best_tc >= 0 and best_worst >= dynamic_threshold:
                profit += (np.sum(y_day[best_td:best_td+8]) - np.sum(y_day[best_tc:best_tc+8])) * 1000

        logger.info(f"  risk_factor={rf} -> 真实验证收益: {profit:.0f}")
        if profit > best_profit:
            best_profit, best_rf = profit, rf

    logger.info(f"  最优: risk_factor={best_rf}, 收益={best_profit:.0f}")

    # 5. 测试集递推预测
    logger.info("[5/6] 测试集递推预测...")

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

    predicted_p10 = np.zeros(len(df_test))
    predicted_p90 = np.zeros(len(df_test))
    predicted_p50 = np.zeros(len(df_test))
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

        p10_pred = model_p10.predict(day_X_with_lag)
        p90_pred = model_p90.predict(day_X_with_lag)
        p50_pred = model_p50.predict(day_X_with_lag)

        predicted_p10[start_idx:end_idx] = p10_pred
        predicted_p90[start_idx:end_idx] = p90_pred
        predicted_p50[start_idx:end_idx] = p50_pred
        all_predicted = np.concatenate([all_predicted, p50_pred])

        if day_idx < 3 or day_idx == n_days - 1:
            logger.info(f"  Day {day_idx+1}: P10={p10_pred.mean():.3f}, P50={p50_pred.mean():.3f}, P90={p90_pred.mean():.3f}")

    df_out = pd.DataFrame({'times': df_test['times'], 'P10': predicted_p10, 'P50': predicted_p50, 'P90': predicted_p90})
    df_out.to_csv(output_price_path, index=False)

    # 6. 生成最终策略
    logger.info("[6/6] 生成最终充放电策略...")
    total_profit = generate_quantile_strategy(
        predicted_p10, predicted_p90, predicted_p50,
        output_power_path, risk_factor=best_rf
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"v38 分位数回归版完成!")
    logger.info(f"测试集总收益: {total_profit:.2f}, 日均: {total_profit/n_days:.2f}")
    logger.info(f"最优risk_factor: {best_rf}")
    logger.info("=" * 60)