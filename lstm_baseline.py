"""
LSTM 电价预测基线：使用 PyTorch LSTM 模型预测节点电价 A
基于时序结构设计，比 GBDT 能更好捕捉时间依赖模式
"""
import pandas as pd
import numpy as np
import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 路径配置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')

output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'lstm_baseline_output.csv')
output_power_path = os.path.join(output_dir, 'output.csv')

# 边界条件特征列（与测试集对齐，仅使用预测值列）
feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'

# LSTM 超参数
SEQ_LEN = 96        # 序列长度：1天 = 96个15分钟时段
HIDDEN_SIZE = 64    # LSTM 隐藏层维度（减小加速）
NUM_LAYERS = 1      # LSTM 层数（减少加速）
DROPOUT = 0.1       # Dropout 比例
BATCH_SIZE = 128    # 批大小（增大加速）
EPOCHS = 20         # 训练轮数（减少加速）
LEARNING_RATE = 0.001  # 学习率
EARLY_STOP_PATIENCE = 5   # 早停耐心值


# ==================== 设备配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")


# ==================== 模型定义 ====================
class LSTMModel(nn.Module):
    """
    LSTM 价格预测模型

    输入： (batch_size, seq_len, input_size)  96 x 11
    输出： (batch_size, 1)
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze(-1)


# ==================== 数据处理函数 ====================
def add_time_features(df):
    """添加时间特征"""
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


def create_sequences(data, seq_len):
    """
    构建时序样本

    参数：
        data: DataFrame，包含特征和目标值
        seq_len: 序列长度

    返回：
        X: numpy array, shape (n_samples, seq_len, n_features)
        y: numpy array, shape (n_samples,)
    """
    X, y = [], []
    # 使用所有列：price + feature_cols + 时间特征
    values = data.values

    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len, 0])  # price 在第一列

    return np.array(X), np.array(y)


def normalize_data(train_data, val_data, test_data):
    """
    对特征进行标准化（仅对特征，不对目标值）

    参数：
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据

    返回：
        标准化后的数据和标准化器
    """
    # 特征标准化
    feature_mean = train_data[:, :, 1:].mean(axis=(0, 1))  # 排除price列
    feature_std = train_data[:, :, 1:].std(axis=(0, 1))

    # 避免除零
    feature_std[feature_std == 0] = 1

    # 标准化特征
    train_data_norm = train_data.copy()
    val_data_norm = val_data.copy()
    test_data_norm = test_data.copy()

    train_data_norm[:, :, 1:] = (train_data[:, :, 1:] - feature_mean) / feature_std
    val_data_norm[:, :, 1:] = (val_data[:, :, 1:] - feature_mean) / feature_std
    test_data_norm[:, :, 1:] = (test_data[:, :, 1:] - feature_mean) / feature_std

    # 目标值标准化
    price_mean = train_data[:, :, 0].mean()
    price_std = train_data[:, :, 0].std()
    price_std = price_std if price_std > 0 else 1

    train_data_norm[:, :, 0] = (train_data[:, :, 0] - price_mean) / price_std
    val_data_norm[:, :, 0] = (val_data[:, :, 0] - price_mean) / price_std
    test_data_norm[:, :, 0] = (test_data[:, :, 0] - price_mean) / price_std

    return train_data_norm, val_data_norm, test_data_norm, price_mean, price_std


# ==================== 训练和推理函数 ====================
def train_model(model, train_loader, val_loader, epochs, lr, patience):
    """训练模型，带早停"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_state = None

    total_batches = len(train_loader)
    logger.info(f"开始训练，总批次数: {total_batches}")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        batch_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1

            # 每10个batch打印一次进度
            if batch_count % 10 == 0:
                print(f"\rEpoch {epoch+1}/{epochs}, Batch {batch_count}/{total_batches}, Loss: {loss.item():.6f}", end="", flush=True)

        print()  # 换行

        train_loss /= total_batches

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.float().to(device)
                batch_y = batch_y.float().to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            logger.info(f"早停：验证损失连续 {patience} 轮未改善")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict(model, X):
    """批量预测"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    return predictions


def recursive_predict(model, initial_seq, n_steps, feature_mean, feature_std, price_mean, price_std):
    """
    递归预测：逐步构建序列进行长序列预测

    参数：
        model: 训练好的模型
        initial_seq: 初始序列 (seq_len, n_features)
        n_steps: 需要预测的步数
        feature_mean, feature_std: 特征标准化参数
        price_mean, price_std: 价格标准化参数

    返回：
        predictions: 预测值数组
    """
    model.eval()
    predictions = []

    # 复制初始序列
    current_seq = initial_seq.copy()

    for _ in range(n_steps):
        # 标准化输入
        input_seq = current_seq.copy()
        input_seq[:, 1:] = (input_seq[:, 1:] - feature_mean) / feature_std
        input_seq[:, 0] = (input_seq[:, 0] - price_mean) / price_std

        # 预测
        X_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_normalized = model(X_tensor).cpu().numpy()[0]

        # 反标准化
        pred = pred_normalized * price_std + price_mean
        predictions.append(pred)

        # 更新序列：用预测值替换最旧的点
        new_point = current_seq[-1].copy()
        new_point[0] = pred  # 更新价格为预测值
        current_seq = np.vstack([current_seq[1:], new_point])

    return np.array(predictions)


# ==================== 充放电策略生成 ====================
def generate_strategy(price_csv, save_path, min_profit_threshold=0):
    """
    根据预测的实时价格确定充放电策略
    """
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])
    df['date'] = df['times'].dt.date

    results = []
    total_profit = 0
    skip_count = 0

    for date, group in df.groupby('date'):
        prices = group['A'].values
        times = group['times'].values

        n = len(prices)
        if n != 96:
            logger.warning(f"{date} 的数据点数量为 {n}, 预期为 96")
            continue

        best_profit = 0
        best_tc = -1
        best_td = -1

        for tc in range(0, 81):
            charge_prices = prices[tc:tc+8]
            charge_cost = np.sum(charge_prices) * 1000

            for td in range(tc + 8, 89):
                discharge_prices = prices[td:td+8]
                discharge_revenue = np.sum(discharge_prices) * 1000
                profit = discharge_revenue - charge_cost

                if profit > best_profit:
                    best_profit = profit
                    best_tc = tc
                    best_td = td

        power = np.zeros(96)
        if best_tc >= 0 and best_td >= 0 and best_profit > min_profit_threshold:
            power[best_tc:best_tc+8] = -1000
            power[best_td:best_td+8] = 1000
            total_profit += best_profit
            logger.info(f"日期: {date}, 充电开始: {best_tc:2d}, 放电开始: {best_td:2d}, 预期收益: {best_profit:10.2f}")
        else:
            skip_count += 1
            logger.info(f"日期: {date}, 跳过交易（收益: {best_profit:.2f} <= 阈值: {min_profit_threshold:.2f}）")

        for i, (t, p, pr) in enumerate(zip(times, power, prices)):
            results.append({'times': t, '实时价格': pr, 'power': p})

    df_result = pd.DataFrame(results)
    df_result.to_csv(save_path, index=False)

    n_days = len(df.groupby("date"))
    avg_profit = total_profit / n_days if n_days > 0 else 0

    logger.info(f"充放电策略已保存: {save_path}")
    logger.info(f"总天数: {n_days}, 执行交易: {n_days - skip_count}, 跳过交易: {skip_count}")
    logger.info(f"总收益: {total_profit:.2f}, 平均日收益: {avg_profit:.2f}")

    return df_result


# ==================== 主程序 ====================
if __name__ == '__main__':
    # 检查数据文件
    if not os.path.exists(train_feature_path):
        logger.error(f"训练特征文件不存在: {train_feature_path}")
        exit(1)
    if not os.path.exists(train_label_path):
        logger.error(f"训练标签文件不存在: {train_label_path}")
        exit(1)
    if not os.path.exists(test_feature_path):
        logger.error(f"测试特征文件不存在: {test_feature_path}")
        exit(1)

    logger.info("=" * 60)
    logger.info("开始训练 LSTM 模型...")
    logger.info("=" * 60)

    # ==================== 1. 数据准备 ====================
    logger.info("[1/4] 加载数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)
    logger.info(f"  训练特征: {df_feat.shape}")
    logger.info(f"  训练标签: {df_label.shape}")

    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    logger.info(f"  合并后: {df_train.shape}")

    df_train = add_time_features(df_train)

    # 准备完整数据用于构建序列
    df_train['price'] = df_train[target_col]
    all_cols = ['price'] + feature_cols + ['hour', 'minute', 'dayofweek', 'month']
    df_train = df_train[all_cols]
    logger.info(f"  特征列: {all_cols}")

    # 按时间顺序划分：80% 训练，20% 验证
    n = len(df_train)
    train_end = int(n * 0.8)

    train_data = df_train.iloc[:train_end].values
    val_data = df_train.iloc[train_end:].values

    logger.info(f"  训练数据: {train_data.shape}, 验证数据: {val_data.shape}")

    # 构建序列
    logger.info(f"  构建序列 (seq_len={SEQ_LEN})...")
    X_train, y_train = create_sequences(pd.DataFrame(train_data, columns=all_cols), SEQ_LEN)
    X_val, y_val = create_sequences(pd.DataFrame(val_data, columns=all_cols), SEQ_LEN)

    logger.info(f"  训练序列: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  验证序列: X={X_val.shape}, y={y_val.shape}")

    # 标准化
    X_train_norm, X_val_norm, _, price_mean, price_std = normalize_data(X_train, X_val, X_val)

    # 创建 DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_norm),
        torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==================== 2. 模型训练 ====================
    logger.info("[2/4] 训练 LSTM 模型...")
    input_size = X_train.shape[2]  # 特征数量
    logger.info(f"  输入特征数: {input_size}")

    model = LSTMModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE)

    # 验证集评估
    y_val_pred = predict(model, X_val_norm)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred * price_std + price_mean))
    mae = mean_absolute_error(y_val, y_val_pred * price_std + price_mean)
    logger.info(f"  验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # ==================== 3. 测试集推理 ====================
    logger.info("[3/4] 测试集推理...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_time_features(df_test)

    # 获取测试集的特征
    test_features = feature_cols + ['hour', 'minute', 'dayofweek', 'month']
    test_data = df_test[test_features].values

    logger.info(f"  测试数据形状: {test_data.shape}")

    # 方法1：直接批量预测（如果有足够的历史数据构建序列）
    # 使用训练集最后96个点 + 测试集构建完整序列
    last_train_seq = df_train[all_cols].iloc[-SEQ_LEN:].values  # (96, n_features)

    # 合并测试数据用于递归预测
    full_seq_for_test = np.vstack([last_train_seq, np.column_stack([
        np.zeros(len(df_test)),  # price 占位
        test_data
    ])])

    # 递归预测测试集
    logger.info("  使用递归预测...")
    predictions = recursive_predict(
        model=model,
        initial_seq=full_seq_for_test[:SEQ_LEN],
        n_steps=len(df_test),
        feature_mean=np.mean(train_data[:, 1:], axis=0),
        feature_std=np.std(train_data[:, 1:], axis=0),
        price_mean=price_mean,
        price_std=price_std
    )

    df_out = pd.DataFrame({'times': df_test['times'], target_col: predictions})
    df_out.to_csv(output_price_path, index=False)
    logger.info(f"  推理结果已保存: {output_price_path}")
    logger.info(f"  预测天数: {len(df_out) // 96} 天")

    # ==================== 4. 生成充放电策略 ====================
    logger.info("[4/4] 生成充放电策略...")
    generate_strategy(output_price_path, output_power_path, min_profit_threshold=0)

    logger.info("=" * 60)
    logger.info("LSTM Baseline 运行完成！")
    logger.info(f"提交文件: {output_power_path}")
    logger.info("=" * 60)
