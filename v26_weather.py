"""
v26 气象数据版 - 在v25基础上整合NWP气象数据
核心升级: 提取ghi(辐照度)和风速(u100/v100)合并到训练集
"""
import pandas as pd
import numpy as np
import os
import netCDF4 as nc
import glob
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')
nc_dir = os.path.join(current_dir, 'data', 'all_nc')

output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'v26_weather_price.csv')
output_power_path = os.path.join(output_dir, 'v26_weather_output.csv')

target_col = 'A'

LAG_96 = 96
LAG_192 = 192


def load_weather_data(nc_dir, start_date, end_date):
    """加载气象数据并转换为15分钟粒度

    NC文件结构: time x lead_time(24h预测) x channel(7个变量) x lat x lon
    channel: ['ghi', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100']
    - ghi: 全天空水平辐射
    - u100/v100: 100米高度风速分量
    """
    nc_files = sorted(glob.glob(os.path.join(nc_dir, '*.nc')))

    weather_list = []
    for nc_file in nc_files:
        filename = os.path.basename(nc_file)
        date_str = filename.replace('.nc', '')

        date = pd.to_datetime(date_str, format='%Y%m%d')

        if date < start_date or date > end_date:
            continue

        ds = nc.Dataset(nc_file, 'r')
        data = ds.variables['data'][:]  # shape: (1, 24, 7, 104, 225)

        # 对空间(lat x lon)做平均，得到24小时 x 7变量的1D序列
        data_mean = data.mean(axis=(3, 4))  # shape: (1, 24, 7)

        # 只取lead_time=0(当前时刻的预测)，避免forecast horizon混淆
        # 或者可以用lead_time平均获得更稳定的特征
        data_mean = data_mean[0, :, :]  # shape: (24, 7)

        # channel索引: 0=ghi, 5=u100, 6=v100
        ghi = data_mean[:, 0]
        u100 = data_mean[:, 5]
        v100 = data_mean[:, 6]

        # 计算综合风速
        wind_speed = np.sqrt(u100**2 + v100**2)

        # 生成24小时时间索引
        day_times = pd.date_range(date, periods=24, freq='h')

        weather_df = pd.DataFrame({
            'times': day_times,
            'ghi': ghi,
            'wind_speed': wind_speed
        })
        weather_list.append(weather_df)
        ds.close()

    if not weather_list:
        return pd.DataFrame()

    df_weather = pd.concat(weather_list, ignore_index=True)

    # 将1小时数据插值到15分钟
    df_weather['times'] = pd.to_datetime(df_weather['times'])
    df_weather = df_weather.set_index('times')
    df_weather_15min = df_weather.resample('15T').interpolate(method='linear').reset_index()
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


def prepare_train_data(df_feat, df_label, df_weather):
    df = pd.merge(df_feat, df_label, on='times', how='inner')
    df['times'] = pd.to_datetime(df['times'])

    # 合并气象数据
    if not df_weather.empty:
        df = pd.merge(df, df_weather, on='times', how='left')
        # 填充缺失的气象数据(如果有的话)
        df['ghi'] = df['ghi'].fillna(df['ghi'].mean())
        df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())

    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_business_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_stats(df, target_col)
    df = df.iloc[LAG_192:].reset_index(drop=True)
    return df


def generate_robust_strategy(price_csv, save_path, min_profit_threshold=0, noise_margin=0.10):
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])
    df['date'] = df['times'].dt.date
    results = []

    nominal_total_profit = 0
    days_operated = 0

    for date, group in df.groupby('date'):
        prices = group['A'].values
        times = group['times'].values
        if len(prices) != 96:
            continue

        best_worst_case_profit = -float('inf')
        best_tc_candidate, best_td_candidate = -1, -1
        best_nominal_profit_for_this_plan = 0

        for tc in range(0, 81):
            base_charge_cost = np.sum(prices[tc:tc+8]) * 1000
            worst_charge_cost = base_charge_cost * (1 + noise_margin)

            for td in range(tc + 8, 89):
                base_discharge_rev = np.sum(prices[td:td+8]) * 1000
                worst_discharge_rev = base_discharge_rev * (1 - noise_margin)

                worst_profit = worst_discharge_rev - worst_charge_cost

                if worst_profit > best_worst_case_profit:
                    best_worst_case_profit = worst_profit
                    best_tc_candidate = tc
                    best_td_candidate = td
                    best_nominal_profit_for_this_plan = base_discharge_rev - base_charge_cost

        power = np.zeros(96)
        if best_tc_candidate >= 0 and best_worst_case_profit >= min_profit_threshold:
            power[best_tc_candidate:best_tc_candidate+8] = -1000
            power[best_td_candidate:best_td_candidate+8] = 1000
            nominal_total_profit += best_nominal_profit_for_this_plan
            days_operated += 1

        results.extend([{'times': t, '实时价格': p, 'power': pw}
                       for t, pw, p in zip(times, power, prices)])

    pd.DataFrame(results).to_csv(save_path, index=False)

    total_days = len(df['date'].unique())
    print(f"  策略执行: 容忍度 ±{noise_margin*100}%, 保底阈值 {min_profit_threshold}")
    print(f"  共 {total_days} 天, 稳健操作 {days_operated} 天")

    return nominal_total_profit


if __name__ == '__main__':
    print("=" * 60)
    print("v26 气象数据版 - 整合NWP气象数据")
    print("=" * 60)

    # ==================== 特征定义 ====================
    base_features = [
        '系统负荷预测值', '风光总加预测值', '联络线预测值',
        '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值'
    ]
    time_features = ['hour', 'minute', 'dayofweek']
    cyclical_features = ['sin_month', 'cos_month', 'sin_hour', 'cos_hour']
    business_features = ['净负荷', '风光渗透率']
    lag_features = ['price_lag_96', 'price_lag_192']
    stat_features = ['rolling_mean', 'rolling_std']
    weather_features = ['ghi', 'wind_speed']

    all_features = base_features + time_features + cyclical_features + business_features + lag_features + stat_features + weather_features

    # ==================== 1. 加载数据 ====================
    print("\n[1/6] 加载数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)

    df_train_feat = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train_feat['times'] = pd.to_datetime(df_train_feat['times'])
    train_start = df_train_feat['times'].min()
    train_end = df_train_feat['times'].max()
    print(f"  训练集时间范围: {train_start} ~ {train_end}")

    # 加载气象数据
    print("  加载气象数据(可能有网络问题请稍候)...")
    try:
        df_weather = load_weather_data(nc_dir, train_start, train_end)
        print(f"  气象数据加载成功: {len(df_weather)} 条")
    except Exception as e:
        print(f"  气象数据加载失败: {e}")
        df_weather = pd.DataFrame()

    df_train = prepare_train_data(df_feat, df_label, df_weather)
    print(f"  训练样本数: {len(df_train)}")

    X = df_train[all_features].values
    y = df_train[target_col].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # ==================== 2. 模型训练 ====================
    print("\n[2/6] 训练GBDT模型...")
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
    print("\n[3/6] 加载测试特征数据...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_time_features(df_test)
    df_test = add_cyclical_features(df_test)
    df_test = add_business_features(df_test)

    # 测试集气象数据(如果存在)
    if not df_weather.empty:
        test_start = df_test['times'].min()
        test_end = df_test['times'].max()
        try:
            df_test_weather = load_weather_data(nc_dir, test_start, test_end)
            df_test = pd.merge(df_test, df_test_weather, on='times', how='left')
            df_test['ghi'] = df_test['ghi'].fillna(df_weather['ghi'].mean())
            df_test['wind_speed'] = df_test['wind_speed'].fillna(df_weather['wind_speed'].mean())
            print(f"  测试集气象数据合并成功")
        except Exception as e:
            print(f"  测试集气象数据加载失败: {e}")
            df_test['ghi'] = 0
            df_test['wind_speed'] = 0
    else:
        df_test['ghi'] = 0
        df_test['wind_speed'] = 0

    print(f"  测试集样本数: {len(df_test)}")

    # ==================== 4. 递推预测 ====================
    print("\n[4/6] 递推预测测试集...")

    train_last_prices = df_train[target_col].values[-LAG_192:]
    print(f"  初始滞后值: shape={train_last_prices.shape}")

    X_test_base = df_test[base_features + time_features + cyclical_features + business_features + weather_features].copy()

    predicted_prices = np.zeros(len(df_test))
    all_predicted = np.concatenate([train_last_prices.copy()])

    n_days = len(df_test) // 96

    for day_idx in range(n_days):
        start_idx = day_idx * 96
        end_idx = start_idx + 96

        day_X = X_test_base.iloc[start_idx:end_idx].values.copy()

        lag_96_arr = all_predicted[-LAG_96:]
        lag_192_arr = all_predicted[-LAG_192:-LAG_96]

        past_192 = all_predicted[-LAG_192:]
        rolling_mean_arr = np.array([np.mean(past_192[i:i+LAG_96]) for i in range(96)])
        rolling_std_arr = np.array([np.std(past_192[i:i+LAG_96]) for i in range(96)])

        lag_features_arr = np.column_stack([lag_96_arr, lag_192_arr])
        stat_arr = np.column_stack([rolling_mean_arr, rolling_std_arr])

        day_X_with_lag = np.concatenate([day_X, lag_features_arr, stat_arr], axis=1)

        day_pred = model.predict(day_X_with_lag)
        predicted_prices[start_idx:end_idx] = day_pred
        all_predicted = np.concatenate([all_predicted, day_pred])

        if day_idx < 5 or day_idx == n_days - 1:
            print(f"  Day {day_idx+1}: mean={day_pred.mean():.4f}")

    # ==================== 5. 保存结果 ====================
    print("\n[5/6] 生成策略...")
    df_out = pd.DataFrame({'times': df_test['times'], target_col: predicted_prices})
    df_out.to_csv(output_price_path, index=False)

    total_profit = generate_robust_strategy(output_price_path, output_power_path, noise_margin=0.0)
    print(f"\n  总收益: {total_profit:.2f}, 日均收益: {total_profit/n_days:.2f}")

    # ==================== 6. 特征重要性 ====================
    print("\n[6/6] 特征重要性:")
    print("=" * 60)
    for feat, imp in sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1]):
        bar = "=" * int(imp * 100)
        print(f"  {feat:<20s}: {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("v26 气象数据版完成!")
    print("=" * 60)