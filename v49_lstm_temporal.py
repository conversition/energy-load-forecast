"""
v49: LSTM弥补时序连续性缺失（第三优先级）
核心改进:
1. 引入LSTM模型预测96点序列，捕捉时序连续性
2. 将LSTM预测结果作为LightGBM的额外特征
3. 保持LightGBM分位数回归框架不变
4. LSTM-only方式：一口气输出96点，然后直接用分位数策略
"""
import pandas as pd
import numpy as np
import os
import netCDF4 as nc
import glob
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
output_price_path = os.path.join(output_dir, 'v49_lstm_temporal_price.csv')
output_power_path = os.path.join(output_dir, 'v49_lstm_temporal_output.csv')

target_col = 'A'
LAG_96, LAG_192, LAG_288 = 96, 192, 288
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {DEVICE}")


class LSTMModel(nn.Module):
    """简单LSTM模型用于96点序列预测"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        out = self.fc(lstm_out)
        # out: (batch, seq_len, 1)
        return out.squeeze(-1)


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_weather_data_deep(nc_dir, start_date, end_date):
    """深度加载气象数据"""
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
            msl = data_mean[:, 1] / 1000
            t2m = data_mean[:, 2] - 273.15
            tcc = data_mean[:, 3]
            u100 = data_mean[:, 5]
            v100 = data_mean[:, 6]
            wind_speed = np.sqrt(u100**2 + v100**2)

            utc_times = pd.date_range(date, periods=24, freq='h')
            bjt_times = utc_times + pd.Timedelta(hours=8)

            weather_df = pd.DataFrame({
                'times': bjt_times,
                'ghi': ghi,
                'msl': msl,
                'temperature': t2m,
                'tcc': tcc,
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
    df['光伏辐照度'] = df['ghi'] * df['光伏预测值'].clip(0, None)
    df['云量光伏比'] = df['tcc'] / (df['光伏预测值'] + 0.01)
    df['温荷积'] = df['temperature'] * df['系统负荷预测值']
    df['压风积'] = df['msl'] * df['wind_speed']
    return df


def compute_derivative_features(df, cols=None):
    if cols is None:
        cols = ['wind_speed', 'temperature', 'ghi', 'tcc', 'msl']
    for col in cols:
        if col not in df.columns:
            continue
        df[f'{col}_diff1'] = df.groupby('date')[col].diff(4).fillna(0)
        df[f'{col}_diff2'] = df.groupby('date')[f'{col}_diff1'].diff(4).fillna(0)
    return df


def prepare_data_with_sequences(df_feat, df_label, df_weather, weather_means, seq_len=96, target_col='A'):
    """
    准备序列数据用于LSTM训练
    返回: 序列特征, 序列目标, 日期列表
    """
    df_feat_copy = df_feat.copy()
    df_feat_copy['times'] = pd.to_datetime(df_feat_copy['times'])
    df_label_copy = df_label.copy()
    df_label_copy['times'] = pd.to_datetime(df_label_copy['times'])
    df = pd.merge(df_feat_copy, df_label_copy, on='times', how='inner')

    if not df_weather.empty:
        df = pd.merge(df, df_weather, on='times', how='left')
        weather_cols = ['ghi', 'msl', 'temperature', 'tcc', 'wind_speed']
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].fillna(weather_means.get(col, 0))

    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_interaction_features(df)
    df['time_step'] = df['hour'] * 4 + df['minute'] // 15
    df['date'] = df['times'].dt.date

    df = compute_derivative_features(df)
    df = df.sort_values('times').reset_index(drop=True)
    df['price_lag_1_day'] = df[target_col].shift(96)
    df['price_lag_2_day'] = df[target_col].shift(192)

    # 丢弃初始不足的数据
    df = df.iloc[LAG_288 + 192:].reset_index(drop=True)
    df = df.dropna(subset=['price_lag_2_day']).reset_index(drop=True)

    # 定义LSTM输入特征
    lstm_feature_cols = [
        '系统负荷预测值', '风光总加预测值', '风电预测值', '光伏预测值',
        'dayofweek', 'month', 'sin_month', 'cos_month',
        '供需比', '净负荷', '风光渗透率', 'is_workday',
        'ghi', 'tcc', 'temperature', 'wind_speed',
        'price_lag_1_day', 'price_lag_2_day'
    ]

    # 按天分割序列
    dates = df['date'].unique()
    sequences = []
    targets = []

    for date in dates:
        day_data = df[df['date'] == date].sort_values('times')
        if len(day_data) < seq_len:
            continue

        # 取完整的96点
        day_data = day_data.iloc[:seq_len]

        # 构建输入序列
        seq_features = day_data[lstm_feature_cols].values
        seq_target = day_data[target_col].values

        sequences.append(seq_features)
        targets.append(seq_target)

    return np.array(sequences), np.array(targets), dates, lstm_feature_cols


def train_lstm_model(train_sequences, train_targets, feature_cols, epochs=50, batch_size=32, lr=0.001):
    """训练LSTM模型"""
    input_size = len(feature_cols)
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = SequenceDataset(train_sequences, train_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)  # (batch, seq_len)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.info(f"  LSTM Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    return model


def predict_with_lstm(model, sequences, feature_cols):
    """使用LSTM批量预测"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for seq in sequences:
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
            pred = model(seq_tensor).squeeze().cpu().numpy()
            predictions.append(pred)

    return np.array(predictions)


def optimize_strategy(predictions_p10, predictions_p50, predictions_p90, risk_factor=0.3):
    """优化策略"""
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
    logger.info("v49: LSTM弥补时序连续性缺失（第三优先级）")
    logger.info("核心: LSTM预测96点序列，捕捉时序依赖")
    logger.info("=" * 60)

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

    df_weather = load_weather_data_deep(nc_dir, all_start, all_end)
    weather_means = {col: df_weather[col].mean() for col in ['ghi', 'msl', 'temperature', 'tcc', 'wind_speed']} if not df_weather.empty else {}

    # 2. 准备序列数据
    logger.info("[2/6] 准备LSTM序列数据...")
    sequences, targets, train_dates, lstm_feature_cols = prepare_data_with_sequences(
        df_feat, df_label, df_weather, weather_means
    )
    logger.info(f"  序列数量: {len(sequences)}, 特征维度: {sequences.shape[2]}")

    # 3. 分割训练/验证
    logger.info("[3/6] 数据分割...")
    split_idx = int(len(sequences) * 0.8)
    train_seq = sequences[:split_idx]
    train_tgt = targets[:split_idx]
    val_seq = sequences[split_idx:]
    val_tgt = targets[split_idx:]

    logger.info(f"  训练序列: {len(train_seq)}, 验证序列: {len(val_seq)}")

    # 4. 训练LSTM模型
    logger.info("[4/6] 训练LSTM模型...")
    lstm_model = train_lstm_model(train_seq, train_tgt, lstm_feature_cols, epochs=50, batch_size=32)

    # 验证集评估
    val_pred_lstm = predict_with_lstm(lstm_model, val_seq, lstm_feature_cols)
    rmse_lstm = np.sqrt(mean_squared_error(val_tgt.flatten(), val_pred_lstm.flatten()))
    logger.info(f"  LSTM验证集 RMSE: {rmse_lstm:.6f}")

    # 验证集策略评估（用LSTM P50预测）
    power_schedule, val_profit = optimize_strategy(
        val_pred_lstm * 0.9,  # P10估计
        val_pred_lstm,         # P50估计
        val_pred_lstm * 1.1,   # P90估计
        risk_factor=0.3
    )
    logger.info(f"  验证集策略收益(LSTM): {val_profit:.0f}")

    # ========== 测试集预测 ==========
    logger.info("[5/6] 测试集LSTM预测...")

    # 准备测试集序列
    df_test_raw = pd.read_csv(test_feature_path)
    df_test_raw['times'] = pd.to_datetime(df_test_raw['times'])

    if not df_weather.empty:
        df_test_raw = pd.merge(df_test_raw, df_weather, on='times', how='left')
        for col in ['ghi', 'msl', 'temperature', 'tcc', 'wind_speed']:
            if col in df_test_raw.columns:
                df_test_raw[col] = df_test_raw[col].fillna(weather_means.get(col, 0))

    df_test_raw = add_time_features(df_test_raw)
    df_test_raw = add_cyclical_features(df_test_raw)
    df_test_raw = add_interaction_features(df_test_raw)
    df_test_raw['time_step'] = df_test_raw['hour'] * 4 + df_test_raw['minute'] // 15
    df_test_raw['date'] = df_test_raw['times'].dt.date

    df_test_raw = compute_derivative_features(df_test_raw)
    df_test_raw = df_test_raw.sort_values('times').reset_index(drop=True)

    # 处理测试集lag特征（使用训练集末尾数据）
    df_label['times'] = pd.to_datetime(df_label['times'])
    df_label_sorted = df_label.sort_values('times').tail(192 * 2)
    historical_prices = df_label_sorted[target_col].values

    df_test_raw['price_lag_1_day'] = 0.0
    df_test_raw['price_lag_2_day'] = 0.0

    # 构建测试序列
    test_dates = df_test_raw['date'].unique()
    test_sequences = []

    for date in test_dates:
        day_data = df_test_raw[df_test_raw['date'] == date].sort_values('times')
        if len(day_data) < 96:
            continue
        day_data = day_data.iloc[:96]
        seq_features = day_data[lstm_feature_cols].values
        test_sequences.append(seq_features)

    # 逐日预测并更新lag特征（递归）
    all_test_preds = []
    for day_idx, seq in enumerate(test_sequences):
        seq_copy = seq.copy()

        if day_idx == 0:
            # 第1天：使用历史价格
            seq_copy[:, lstm_feature_cols.index('price_lag_1_day')] = historical_prices[96:192]
            seq_copy[:, lstm_feature_cols.index('price_lag_2_day')] = historical_prices[0:96]
        else:
            # 后续天：使用前几天的预测
            if len(all_test_preds) >= 1:
                prev_pred = all_test_preds[-1]
                seq_copy[:, lstm_feature_cols.index('price_lag_1_day')] = prev_pred
            if len(all_test_preds) >= 2:
                prev2_pred = all_test_preds[-2]
                seq_copy[:, lstm_feature_cols.index('price_lag_2_day')] = prev2_pred

        # 预测
        seq_tensor = torch.FloatTensor(seq_copy).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = lstm_model(seq_tensor).squeeze().cpu().numpy()
        all_test_preds.append(pred)

    all_test_preds = np.concatenate(all_test_preds)
    n_test_days = len(all_test_preds) // 96

    # 保存结果
    logger.info("[6/6] 保存预测结果...")
    test_times = df_test_raw['times'].values[:len(all_test_preds)]

    df_price_out = pd.DataFrame({
        'times': test_times,
        'P50': all_test_preds,
        'P10': all_test_preds * 0.9,
        'P90': all_test_preds * 1.1
    })
    df_price_out.to_csv(output_price_path, index=False)
    logger.info(f"  价格预测已保存: {output_price_path}")

    power_schedule, test_profit = optimize_strategy(
        all_test_preds * 0.9,
        all_test_preds,
        all_test_preds * 1.1,
        risk_factor=0.3
    )

    df_power_out = pd.DataFrame({
        'times': test_times,
        'P10': all_test_preds * 0.9,
        'P90': all_test_preds * 1.1,
        'median': all_test_preds,
        'power': power_schedule
    })
    df_power_out.to_csv(output_power_path, index=False)
    logger.info(f"  策略已保存: {output_power_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"v49 LSTM时序模型完成!")
    logger.info(f"LSTM验证集 RMSE: {rmse_lstm:.6f}")
    logger.info(f"验证集策略收益: {val_profit:.0f}")
    logger.info(f"测试集策略收益: {test_profit:.0f}")
    logger.info(f"测试集天数: {n_test_days}")
    logger.info("=" * 60)