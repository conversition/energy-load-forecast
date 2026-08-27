"""
v45 Physics Constraints + Derivative Features（里程碑2）
核心改进：
1. monotone_constraints: 物理约束注入
   - 净负荷与电价正相关 (+1)
   - 风光渗透率与电价负相关 (-1)
2. 导数特征(按日隔离): groupby('date').diff()
3. 顶峰压力指数: exp(净负荷/历史最大负荷)
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
    """时间特征"""
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


def add_cyclical_features(df):
    """周期特征"""
    df = df.copy()
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


def add_interaction_features(df):
    """交互特征"""
    df = df.copy()
    df['供需比'] = df['风光总加预测值'] / (df['系统负荷预测值'] + 1)
    df['风光差'] = df['风电预测值'] - df['光伏预测值']
    df['净负荷'] = df['系统负荷预测值'] - df['风光总加预测值']
    total_gen = df['风光总加预测值'] + df['水电预测值'] + df['非市场化机组预测值']
    df['风光渗透率'] = df['风光总加预测值'] / (total_gen + 1)
    df['is_workday'] = (df['dayofweek'] < 5).astype(int)
    return df


def compute_derivative_features(df, cols=['wind_speed', 'temperature']):
    """
    计算气象变量的时序导数
    按 date 分组隔离，避免跨日产生无意义的极值
    """
    for col in cols:
        if col not in df.columns:
            continue
        # 一阶差分：4个15分钟=1小时前的变化
        df[f'{col}_diff1'] = df.groupby('date')[col].diff(4).fillna(0)
        # 二阶差分：加速度
        df[f'{col}_diff2'] = df.groupby('date')[f'{col}_diff1'].diff(4).fillna(0)
    return df


def compute_pressure_index(df):
    """
    顶峰压力指数: exp(净负荷/历史最大负荷)
    当净负荷逼近传统机组极限出力时，价格呈指数级飙升
    """
    max_load = df['系统负荷预测值'].max()
    df['顶峰压力指数'] = np.exp(df['净负荷'] / max_load)
    df['顶峰压力线性'] = df['净负荷'] / max_load
    return df


def compute_deviation_features(df_feat, deviation_cols):
    """计算偏差特征"""
    df_feat_out = df_feat.copy()
    df_feat_out['times'] = pd.to_datetime(df_feat_out['times'])
    df_feat_out = add_time_features(df_feat_out)

    deviation_stats = {}
    for col in deviation_cols:
        actual_col = f'{col}实际值'
        pred_col = f'{col}预测值'
        if actual_col in df_feat_out.columns and pred_col in df_feat_out.columns:
            deviation_col = f'{col}_deviation'
            df_feat_out[deviation_col] = df_feat_out[actual_col] - df_feat_out[pred_col]
            grouped_mean = df_feat_out.groupby(['hour', 'month'])[deviation_col].mean()
            deviation_stats[col] = grouped_mean.to_dict()

    for col in deviation_cols:
        if col in deviation_stats:
            df_feat_out[f'{col}_dev_mean'] = df_feat_out.apply(
                lambda row: deviation_stats[col].get((row['hour'], row['month']), 0), axis=1
            )

    return df_feat_out, deviation_stats


def get_feature_constraints(feature_names):
    """
    获取 monotone_constraints 数组
    物理约束：
    - 净负荷与电价正相关 (+1)
    - 风光渗透率与电价负相关 (-1)
    """
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


def prepare_time_of_day_data(df_feat, df_label, df_weather, weather_means):
    """准备时刻专属模型的数据格式"""
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

    # 时刻标识（需要在导数计算之前）
    df['time_step'] = df['hour'] * 4 + df['minute'] // 15
    df['date'] = df['times'].dt.date

    # 导数特征（按日隔离）- 需要在添加date之后
    df = compute_derivative_features(df, cols=['wind_speed', 'temperature'])

    # 顶峰压力指数
    df = compute_pressure_index(df)

    df = df.iloc[LAG_288:].reset_index(drop=True)

    return df, df['date'].unique()


class PhysicsConstrainedPredictor:
    """带物理约束的96个时刻专属模型"""

    def __init__(self, n_steps=96):
        self.models = {}
        self.n_steps = n_steps

    def train(self, df_train, feature_cols, target_col):
        """训练带单调性约束的96个时刻专属模型"""
        constraints = get_feature_constraints(feature_cols)

        for step in range(self.n_steps):
            df_step = df_train[df_train['time_step'] == step].copy()

            if len(df_step) < 10:
                continue

            X_step = df_step[feature_cols].values
            y_step = df_step[target_col].values

            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=7,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                monotone_constraints=constraints,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_step, y_step)
            self.models[step] = model

            if step % 24 == 0:
                logger.info(f"  已训练时刻模型: {step + 1}/96")

        logger.info(f"  共训练 {len(self.models)} 个带物理约束的时刻专属模型")

    def predict(self, df_test, feature_cols):
        """预测测试集"""
        predictions = np.zeros(len(df_test))

        for step in range(self.n_steps):
            step_mask = df_test['time_step'] == step

            if not step_mask.any():
                continue

            X_test_step = df_test.loc[step_mask, feature_cols].values

            if step in self.models:
                predictions[step_mask] = self.models[step].predict(X_test_step)

        return predictions


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("v45 Physics Constraints + Derivative Features (里程碑2)")
    logger.info("核心: monotone_constraints + 导数特征 + 顶峰压力指数")
    logger.info("=" * 60)

    # 特征配置
    base_features = ['系统负荷预测值', '风光总加预测值', '联络线预测值', '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
    time_features = ['hour', 'minute', 'dayofweek']
    cyclical_features = ['sin_month', 'cos_month', 'sin_hour', 'cos_hour']
    physics_features = ['供需比', '风光差', '净负荷', '风光渗透率', 'is_workday', '顶峰压力指数', '顶峰压力线性']
    weather_features = ['wind_speed', 'temperature']
    derivative_features = ['wind_speed_diff1', 'wind_speed_diff2', 'temperature_diff1', 'temperature_diff2']

    deviation_cols = ['系统负荷', '风光总加', '风电', '光伏']
    deviation_feature_names = [f'{col}_dev_mean' for col in deviation_cols]

    all_features = base_features + time_features + cyclical_features + physics_features + weather_features + derivative_features + deviation_feature_names

    logger.info(f"特征数量: {len(all_features)}")
    constraints = get_feature_constraints(all_features)
    logger.info(f"单调性约束: 净负荷→+1, 风光渗透率→-1")

    # 1. 加载数据
    logger.info("[1/5] 加载数据...")
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

    # 2. 计算偏差特征
    logger.info("[2/5] 计算偏差特征...")
    df_feat_with_dev, deviation_stats = compute_deviation_features(df_feat, deviation_cols)

    # 3. 准备数据
    logger.info("[3/5] 准备带物理约束的时刻专属模型数据...")
    df_train, train_dates = prepare_time_of_day_data(df_feat_with_dev, df_label, df_weather, weather_means)

    split_idx = int(len(df_train) * 0.8)
    df_train_split = df_train.iloc[:split_idx]
    df_val_split = df_train.iloc[split_idx:]

    y_train_full = df_train_split[target_col].values
    y_val_full = df_val_split[target_col].values

    logger.info(f"训练样本: {len(df_train_split)}, 验证样本: {len(df_val_split)}")

    # 4. 训练带物理约束的模型
    logger.info("[4/5] 训练带物理约束的96个时刻专属模型...")
    predictor = PhysicsConstrainedPredictor(n_steps=96)
    predictor.train(df_train_split, all_features, target_col)

    # 5. 验证集评估
    logger.info("[5/5] 验证集评估...")
    y_val_pred = predictor.predict(df_val_split, all_features)

    rmse = np.sqrt(mean_squared_error(y_val_full, y_val_pred))
    mae = np.mean(np.abs(y_val_full - y_val_pred))

    logger.info(f"  验证集 RMSE: {rmse:.6f}")
    logger.info(f"  验证集 MAE: {mae:.6f}")

    # 分天数计算模拟收益
    n_val_days = len(y_val_pred) // 96
    total_profit = 0
    for day_idx in range(n_val_days):
        start = day_idx * 96
        end = start + 96
        y_day_pred = y_val_pred[start:end]
        y_day_true = y_val_full[start:end]

        best_tc, best_td = -1, -1
        best_profit = -np.inf

        for tc in range(0, 81):
            charge_cost = np.sum(y_day_pred[tc:tc+8])
            for td in range(tc + 8, 89):
                discharge_rev = np.sum(y_day_pred[td:td+8])
                profit = discharge_rev - charge_cost
                if profit > best_profit:
                    best_profit = profit
                    best_tc, best_td = tc, td

        if best_tc >= 0:
            total_profit += (np.sum(y_day_true[best_td:best_td+8]) - np.sum(y_day_true[best_tc:best_tc+8])) * 1000

    logger.info(f"  验证集模拟收益: {total_profit:.0f}")

    # 保存验证集预测
    df_val_out = pd.DataFrame({
        'times': df_val_split['times'].values,
        'predicted': y_val_pred,
        'actual': y_val_full
    })
    df_val_out.to_csv(output_dir + '/v45_val_predictions.csv', index=False)

    logger.info("\n" + "=" * 60)
    logger.info(f"v45 Physics Constraints 完成!")
    logger.info(f"验证集 RMSE: {rmse:.6f}")
    logger.info(f"验证集模拟收益: {total_profit:.0f}")
    logger.info("=" * 60)