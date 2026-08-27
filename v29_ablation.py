"""
v29 实验版 - 气象相关性分析与对照组实验(Ablation Study)
基于v28添加:
1. 气象特征相关性热力图分析
2. 自动化对照组实验
"""
import pandas as pd
import numpy as np
import os
import netCDF4 as nc
import glob
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')
nc_dir = os.path.join(current_dir, 'data', 'all_nc')

output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'v29_exp_price.csv')
output_power_path = os.path.join(output_dir, 'v29_exp_output.csv')

target_col = 'A'

LAG_96 = 96
LAG_192 = 192
LAG_288 = 288


def load_weather_data_fixed(nc_dir, start_date, end_date):
    """修复版气象数据加载"""
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
            data = ds.variables['data'][:]
            data_mean = data.mean(axis=(3, 4))[0, :, :]

            ghi = data_mean[:, 0]
            t2m = data_mean[:, 2] - 273.15
            u100 = data_mean[:, 5]
            v100 = data_mean[:, 6]
            wind_speed = np.sqrt(u100**2 + v100**2)

            utc_times = pd.date_range(date, periods=24, freq='h')

            weather_df = pd.DataFrame({
                'times': utc_times,
                'temperature': t2m,
                'ghi': ghi,
                'wind_speed': wind_speed
            })
            weather_list.append(weather_df)
            ds.close()
        except:
            continue

    if not weather_list:
        return pd.DataFrame()

    df_weather = pd.concat(weather_list, ignore_index=True)
    df_weather['times'] = pd.to_datetime(df_weather['times'])
    df_weather = df_weather.set_index('times')
    df_weather_15min = df_weather.resample('15min').interpolate(method='quadratic').reset_index()
    df_weather_15min = df_weather_15min.rename(columns={'index': 'times'})

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


def add_lag_features(df, price_col):
    df = df.copy()
    df['price_lag_96'] = df[price_col].shift(LAG_96)
    df['price_lag_192'] = df[price_col].shift(LAG_192)
    return df


def add_rolling_stats(df, price_col):
    df = df.copy()
    rolling = df['price_lag_96'].rolling(window=LAG_96, min_periods=1)
    df['rolling_mean'] = rolling.mean()
    df['rolling_std'] = rolling.std()
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
    df = add_lag_features(df, target_col)
    df = add_rolling_stats(df, target_col)
    df = df.iloc[LAG_192:].reset_index(drop=True)
    return df


def plot_weather_correlation(df_train, target_col='A'):
    """气象特征相关性分析（打印方式）"""
    cols_to_analyze = ['temperature', 'ghi', 'wind_speed', '系统负荷预测值', target_col]
    corr_matrix = df_train[cols_to_analyze].corr()

    logger.info("\n========== 气象特征相关性矩阵 ==========")
    logger.info("\n" + corr_matrix.to_string())

    # 保存到CSV
    corr_matrix.to_csv('output/weather_correlation.csv')
    logger.info("相关性矩阵已保存至 output/weather_correlation.csv")


def run_ablation_study(df_train, y, split_idx, experiment_configs, base_features,
                       time_features, cyclical_features, business_features,
                       lag_features, stat_features):
    """自动化对照组实验 (Ablation Study)"""
    logger.info("\n========== 开始自动化对照组实验 ==========")
    results_log = []

    for exp_name, added_features in experiment_configs.items():
        current_features = (base_features + time_features + cyclical_features +
                           business_features + lag_features + stat_features + added_features)

        try:
            X_exp = df_train[current_features].values
            X_train_exp = X_exp[:split_idx]
            X_val_exp = X_exp[split_idx:]
            y_train_exp = y[:split_idx]
            y_val_exp = y[split_idx:]

            model_exp = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42
            )
            model_exp.fit(X_train_exp, y_train_exp)
            y_val_pred_exp = model_exp.predict(X_val_exp)

            rmse_exp = np.sqrt(mean_squared_error(y_val_exp, y_val_pred_exp))
            results_log.append({
                'Experiment': exp_name,
                'RMSE': rmse_exp,
                'Feature_Count': len(current_features)
            })
            logger.info(f"[{exp_name}] -> 验证集 RMSE: {rmse_exp:.5f}")
        except Exception as e:
            logger.warning(f"[{exp_name}] 实验失败: {e}")

    exp_df = pd.DataFrame(results_log).sort_values(by='RMSE')
    logger.info("\n========== 实验结果排行榜 ==========")
    logger.info("\n" + exp_df.to_string(index=False))
    return exp_df


def generate_strategy(price_csv, save_path, noise_margin=0.0, threshold=0):
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])
    df['date'] = df['times'].dt.date
    results = []
    total_profit = 0
    days_operated = 0

    for date, group in df.groupby('date'):
        prices = group['A'].values
        times = group['times'].values
        if len(prices) != 96:
            continue

        best_worst = -float('inf')
        best_tc, best_td = -1, -1

        for tc in range(0, 81):
            base_charge = np.sum(prices[tc:tc+8]) * 1000
            worst_charge = base_charge * (1 + noise_margin)

            for td in range(tc + 8, 89):
                base_dis = np.sum(prices[td:td+8]) * 1000
                worst_dis = base_dis * (1 - noise_margin)
                worst_profit = worst_dis - worst_charge

                if worst_profit > best_worst:
                    best_worst = worst_profit
                    best_tc, best_td = tc, td

        power = np.zeros(96)
        if best_tc >= 0 and best_worst >= threshold:
            power[best_tc:best_tc+8] = -1000
            power[best_td:best_td+8] = 1000
            nominal = (np.sum(prices[best_td:best_td+8]) - np.sum(prices[best_tc:best_tc+8])) * 1000
            total_profit += nominal
            days_operated += 1

        results.extend([{'times': t, '实时价格': p, 'power': pw}
                       for t, pw, p in zip(times, power, prices)])

    pd.DataFrame(results).to_csv(save_path, index=False)
    logger.info(f"策略: noise={noise_margin}, threshold={threshold}, 操作{days_operated}天")
    return total_profit


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("v29 实验版 - 气象相关性分析 + Ablation Study")
    logger.info("=" * 60)

    # ==================== 基础特征定义 ====================
    base_features = [
        '系统负荷预测值', '风光总加预测值', '联络线预测值',
        '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值'
    ]
    time_features = ['hour', 'minute', 'dayofweek']
    cyclical_features = ['sin_month', 'cos_month', 'sin_hour', 'cos_hour']
    business_features = ['净负荷', '风光渗透率']
    lag_features = ['price_lag_96', 'price_lag_192']
    stat_features = ['rolling_mean', 'rolling_std']
    weather_features = ['temperature', 'ghi', 'wind_speed']

    # ==================== 1. 加载数据 ====================
    logger.info("[1/7] 加载数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)
    df_test = pd.read_csv(test_feature_path)

    df_train_feat = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train_feat['times'] = pd.to_datetime(df_train_feat['times'])
    df_test['times'] = pd.to_datetime(df_test['times'])

    train_start = df_train_feat['times'].min()
    train_end = df_train_feat['times'].max()
    test_start = df_test['times'].min()
    test_end = df_test['times'].max()
    all_start = min(train_start, test_start)
    all_end = max(train_end, test_end)

    # ==================== 2. 加载气象数据 ====================
    logger.info("[2/7] 加载气象数据...")
    df_weather = load_weather_data_fixed(nc_dir, all_start, all_end)
    weather_means = {}
    if not df_weather.empty:
        for col in ['temperature', 'ghi', 'wind_speed']:
            weather_means[col] = df_weather[col].mean()
        logger.info(f"气象全局均值: {weather_means}")

    # ==================== 3. 准备训练数据 ====================
    logger.info("[3/7] 准备训练数据...")
    df_train = prepare_train_data(df_feat, df_label, df_weather, weather_means)
    logger.info(f"训练样本数: {len(df_train)}")

    y = df_train[target_col].values
    split_idx = int(len(y) * 0.8)
    y_train, y_val = y[:split_idx], y[split_idx:]

    # ==================== 4. 相关性热力图分析 ====================
    logger.info("[4/7] 生成气象特征相关性热力图...")
    plot_weather_correlation(df_train)

    # ==================== 5. 对照组实验 (Ablation Study) ====================
    logger.info("[5/7] 执行对照组实验...")

    experiment_configs = {
        "Baseline (无气象)": [],
        "+ 仅光照 (ghi)": ['ghi'],
        "+ 仅风速 (wind)": ['wind_speed'],
        "+ 仅温度 (temp)": ['temperature'],
        "+ 光照+风速": ['ghi', 'wind_speed'],
        "+ 风速+温度": ['wind_speed', 'temperature'],
        "+ 光照+温度": ['ghi', 'temperature'],
        "All Weather (全量)": ['temperature', 'ghi', 'wind_speed']
    }

    exp_df = run_ablation_study(
        df_train, y, split_idx, experiment_configs,
        base_features, time_features, cyclical_features,
        business_features, lag_features, stat_features
    )

    # 找出最优实验配置
    best_exp = exp_df.iloc[0]['Experiment']
    logger.info(f"\n最优实验: {best_exp}")

    # ==================== 6. 使用最优配置训练完整模型 ====================
    logger.info("[6/7] 使用最优配置训练完整模型...")

    # 根据Ablation结果决定使用哪些气象特征
    if best_exp == "All Weather (全量)":
        final_weather_features = weather_features
    elif best_exp == "+ 光照+温度":
        final_weather_features = ['ghi', 'temperature']
    elif best_exp == "+ 风速+温度":
        final_weather_features = ['wind_speed', 'temperature']
    elif best_exp == "+ 光照+风速":
        final_weather_features = ['ghi', 'wind_speed']
    elif best_exp == "+ 仅温度 (temp)":
        final_weather_features = ['temperature']
    elif best_exp == "+ 仅风速 (wind)":
        final_weather_features = ['wind_speed']
    elif best_exp == "+ 仅光照 (ghi)":
        final_weather_features = ['ghi']
    else:
        final_weather_features = []  # Baseline

    all_features = (base_features + time_features + cyclical_features +
                   business_features + lag_features + stat_features + final_weather_features)
    logger.info(f"最终特征配置 ({len(all_features)}个): {all_features}")

    X = df_train[all_features].values
    X_train, X_val = X[:split_idx], X[split_idx:]

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, verbose=0
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    logger.info(f"验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # ==================== 7. 测试集递推预测 ====================
    logger.info("[7/7] 测试集递推预测...")

    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])

    if not df_weather.empty:
        df_test = pd.merge(df_test, df_weather, on='times', how='left')
        for col in ['temperature', 'ghi', 'wind_speed']:
            df_test[col] = df_test[col].fillna(weather_means.get(col, 0))

    df_test = add_time_features(df_test)
    df_test = add_cyclical_features(df_test)
    df_test = add_business_features(df_test)

    train_last_prices = df_train[target_col].values[-LAG_192:].copy()
    X_test_base = df_test[base_features + time_features + cyclical_features +
                         business_features + final_weather_features].copy()

    predicted_prices = np.zeros(len(df_test))
    all_predicted = np.concatenate([train_last_prices.copy()])

    n_days = len(df_test) // 96
    logger.info(f"测试集天数: {n_days}")

    for day_idx in range(n_days):
        start_idx = day_idx * 96
        end_idx = start_idx + 96

        day_X = X_test_base.iloc[start_idx:end_idx].values.copy()

        history_series = pd.Series(all_predicted[-LAG_288:])

        lag_96_series = history_series.shift(LAG_96)
        lag_192_series = history_series.shift(LAG_192)
        roll_mean_series = lag_96_series.rolling(LAG_96, min_periods=1).mean()
        roll_std_series = lag_96_series.rolling(LAG_96, min_periods=1).std()

        lag_96_arr = lag_96_series.values[-LAG_96:]
        lag_192_arr = lag_192_series.values[-LAG_96:]
        rolling_mean_arr = roll_mean_series.values[-LAG_96:]
        rolling_std_arr = roll_std_series.values[-LAG_96:]

        overall_mean = np.nanmean(all_predicted)
        overall_std = np.nanstd(all_predicted)
        lag_96_arr = np.nan_to_num(lag_96_arr, nan=overall_mean)
        lag_192_arr = np.nan_to_num(lag_192_arr, nan=overall_mean)
        rolling_mean_arr = np.nan_to_num(rolling_mean_arr, nan=overall_mean)
        rolling_std_arr = np.nan_to_num(rolling_std_arr, nan=overall_std)

        lag_features_arr = np.column_stack([lag_96_arr, lag_192_arr])
        stat_arr = np.column_stack([rolling_mean_arr, rolling_std_arr])

        day_X_with_lag = np.concatenate([day_X, lag_features_arr, stat_arr], axis=1)

        day_pred = model.predict(day_X_with_lag)
        predicted_prices[start_idx:end_idx] = day_pred
        all_predicted = np.concatenate([all_predicted, day_pred])

        if day_idx < 3 or day_idx == n_days - 1:
            logger.info(f"  Day {day_idx+1}: mean={day_pred.mean():.4f}")

    df_out = pd.DataFrame({'times': df_test['times'], target_col: predicted_prices})
    df_out.to_csv(output_price_path, index=False)

    total_profit = generate_strategy(output_price_path, output_power_path)
    logger.info(f"\n总收益: {total_profit:.2f}, 日均: {total_profit/n_days:.2f}")

    logger.info("\n" + "=" * 60)
    logger.info("Feature Importance (v29 实验版):")
    for feat, imp in sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1]):
        bar = "=" * int(imp * 100)
        logger.info(f"  {feat:<20s}: {imp:.4f} {bar}")

    logger.info("v29 实验版完成!")