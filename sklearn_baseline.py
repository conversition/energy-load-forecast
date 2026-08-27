"""
Sklearn GradientBoosting 基线：根据边界条件预测节点电价 A
（纯sklearn实现，无需LightGBM/XGBoost）
"""
import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import netCDF4 as nc
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False
    logging.warning("netCDF4 未安装，气象特征将不可用。请运行: pip install netCDF4")

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 路径配置 ====================
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 数据目录（请根据实际下载的数据位置修改）
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')

# 输出目录
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'sklearn_baseline_output.csv')
output_power_path = os.path.join(output_dir, 'output.csv')  # 最终提交文件

# 边界条件特征列（与测试集对齐，仅使用预测值列）
feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'


# 添加时间特征
def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    # 周期性编码：month 用 sin/cos 表示，解决训练/测试分布偏移问题
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


# 添加交互特征
def add_interaction_features(df):
    """
    添加交互特征，增强模型表达能力
    - 风光比：风电/光伏（反映风光资源配比）
    - 负荷风光差：系统负荷 - 风光总加（电网供需平衡指标）
    - 水电占比：水电 / 系统负荷（水电对电网的贡献度）
    """
    df = df.copy()
    # 风光比
    df['风光比'] = df['风电预测值'] / (df['光伏预测值'] + 1)
    # 负荷风光差
    df['负荷风光差'] = df['系统负荷预测值'] - df['风光总加预测值']
    # 水电占比
    df['水电占比'] = df['水电预测值'] / (df['系统负荷预测值'] + 1)
    return df


# ==================== 气象数据加载(NWP特征) ====================
def inspect_nc_file(nc_path):
    """
    检查 .nc 文件内容，列出所有变量和维度信息

    参数：
        nc_path: .nc 文件路径

    返回：
        变量名列表和维度信息
    """
    if not HAS_NETCDF:
        logger.error("netCDF4 库未安装，无法读取 .nc 文件")
        return None, None

    dataset = nc.Dataset(nc_path, 'r')
    variables = list(dataset.variables.keys())
    dimensions = dict(dataset.dimensions)
    logger.info(f"文件: {nc_path}")
    logger.info(f"维度: {dimensions}")
    logger.info(f"变量: {variables}")
    dataset.close()
    return variables, dimensions


def load_nwp_features(nc_dir, times_index, intervals_per_hour=4):
    """
    从 .nc 文件加载气象特征（风速、辐照度等），做空间平均并对齐到训练数据时间

    nc文件维度：(time=1, lead_time=24, channel=7, lat=104, lon=225)
    - time: 预报时间（通常=1）
    - lead_time: 预报时效（24小时，从0点到23点）
    - channel: 7个通道，对应不同气象变量
    - lat/lon: 空间网格

    参数：
        nc_dir: 气象数据目录（包含 .nc 文件）
        times_index: pandas DatetimeIndex，需对齐的时间索引
        intervals_per_hour: 每小时的时间段数，默认 4（15分钟）

    返回：
        DataFrame，含气象特征列
    """
    if not HAS_NETCDF:
        logger.warning("netCDF4 未安装，跳过气象特征加载")
        return None

    import glob
    from datetime import datetime

    nc_files = sorted(glob.glob(os.path.join(nc_dir, '*.nc')))
    if not nc_files:
        logger.warning(f"气象数据目录 {nc_dir} 中未找到 .nc 文件")
        return None

    logger.info(f"找到 {len(nc_files)} 个气象数据文件")

    # 存储所有日期的气象数据
    # 每天的 .nc 文件包含 24 小时 * 7 个 channel 的数据
    all_data = {}  # {date: {channel: spatial_avg_24h}}

    for nc_path in nc_files:
        try:
            # 从文件名提取日期（如 20250101.nc -> 2025-01-01）
            filename = os.path.basename(nc_path)
            date_str = filename.replace('.nc', '')
            try:
                file_date = datetime.strptime(date_str, '%Y%m%d')
            except:
                logger.warning(f"无法解析文件名日期: {filename}")
                continue

            dataset = nc.Dataset(nc_path, 'r')
            data = dataset.variables['data'][:]  # shape: (1, 24, 7, 104, 225)

            # data 维度: (time, lead_time, channel, lat, lon)
            # 提取每个 channel 在 24 小时的空间平均
            for ch in range(7):
                channel_data = data[0, :, ch, :, :]  # shape: (24, 104, 225)
                # 空间平均
                spatial_avg = np.mean(channel_data, axis=(-2, -1))  # shape: (24,)
                date_key = file_date.strftime('%Y-%m-%d')
                if date_key not in all_data:
                    all_data[date_key] = {}
                all_data[date_key][ch] = spatial_avg

            dataset.close()

        except Exception as e:
            logger.error(f"处理文件 {nc_path} 时出错: {e}")
            continue

    if not all_data:
        logger.warning("未能从气象数据中提取任何特征")
        return None

    logger.info(f"已处理 {len(all_data)} 天的气象数据")

    # 准备输出数据
    n_target = len(times_index)
    n_channels = 7
    nwp_features = np.zeros((n_target, n_channels))

    for i, ts in enumerate(times_index):
        date_key = ts.strftime('%Y-%m-%d')
        if date_key in all_data:
            # 当天的小时数（0-23）
            hour = ts.hour
            # 当天的分钟数转换为小时分数
            minute_fraction = ts.minute / 60.0
            hour_idx = hour + minute_fraction  # 如 14:15 -> 14.25

            for ch in range(n_channels):
                channel_24h = all_data[date_key][ch]  # shape: (24,)
                # 线性插值到当前时间点
                h_low = min(int(hour_idx), 22)  # 确保 h_low 最多 22（这样 h_high 最多 23）
                h_high = h_low + 1
                weight = hour_idx - h_low
                nwp_features[i, ch] = (1 - weight) * channel_24h[h_low] + weight * channel_24h[h_high]
        else:
            # 如果当天没有数据，使用均值填充
            logger.debug(f"日期 {date_key} 没有气象数据，使用均值填充")
            for ch in range(n_channels):
                # 计算所有数据的均值
                all_vals = [all_data[d][ch].mean() for d in all_data if all_data[d][ch].size > 0]
                nwp_features[i, ch] = np.mean(all_vals) if all_vals else 0

    # 创建 DataFrame
    channel_names = [f'nwp_ch{i}' for i in range(n_channels)]
    df_nwp = pd.DataFrame(nwp_features, index=times_index, columns=channel_names)
    logger.info(f"气象特征加载完成，形状: {df_nwp.shape}")
    logger.info(f"  各通道统计:")
    for i, col in enumerate(df_nwp.columns):
        logger.info(f"    {col}: mean={df_nwp[col].mean():.2f}, std={df_nwp[col].std():.2f}")

    return df_nwp


# ==================== 递推预测函数 ====================
def recursive_predict_with_lag(model, df_train, df_test, base_features, lag_features, nwp_features, target_col):
    """
    递推预测：利用滞后特征进行递归预测

    参数：
        model: 训练好的模型
        df_train: 训练数据（含真实电价）
        df_test: 测试数据（不含电价）
        base_features: 基础特征列表
        lag_features: 滞后特征列表
        nwp_features: 气象特征列表
        target_col: 目标列名

    返回：
        predictions: 预测值数组
    """
    logger.info("  使用递推预测构建滞后特征...")

    # 获取训练集最后96点真实电价（用于初始化滞后特征）
    last_prices = df_train[target_col].iloc[-96:].values
    logger.info(f"  初始化滞后特征：使用训练集最后96点电价")

    predictions = []

    # 复制测试数据以构建特征
    test_data = df_test.copy()

    # 存储预测值用于更新滞后
    predicted_prices = list(last_prices)

    # 所有特征列表顺序
    all_features = base_features + lag_features + nwp_features

    for i in range(len(test_data)):
        # 构建当前样本的特征
        row_features = {}

        # 基础特征
        for feat in base_features:
            row_features[feat] = test_data.iloc[i][feat]

        # 滞后特征：使用预测值递推构建
        for lag_feat in lag_features:
            lag_idx = int(lag_feat.split('_')[-1])  # 从 price_lag_96 提取 96
            idx_in_predicted = i - lag_idx + len(last_prices)
            if idx_in_predicted >= 0 and idx_in_predicted < len(predicted_prices):
                row_features[lag_feat] = predicted_prices[idx_in_predicted]
            else:
                hist_idx = i - lag_idx
                if hist_idx < 0 and hist_idx >= -len(last_prices):
                    row_features[lag_feat] = last_prices[hist_idx]
                else:
                    row_features[lag_feat] = np.mean(last_prices)

        # 气象特征
        for nwp_feat in nwp_features:
            row_features[nwp_feat] = test_data.iloc[i][nwp_feat]

        # 构建特征向量
        feature_vector = [row_features[f] for f in all_features]
        X_pred = np.array(feature_vector).reshape(1, -1)

        # 预测
        pred = model.predict(X_pred)[0]
        predictions.append(pred)

        # 更新预测值列表（用于下一个点的滞后特征）
        predicted_prices.append(pred)

        # 进度显示
        if (i + 1) % 500 == 0:
            logger.info(f"    预测进度: {i+1}/{len(test_data)}")

    logger.info(f"  递推预测完成，共 {len(predictions)} 个预测值")
    return np.array(predictions)


# ==================== 充放电策略生成 ====================
# v1_dynamic 策略参数：可变时长范围
# 竞赛规则要求：充放电时段必须恰好是连续2小时（8个时间点）
MIN_DURATION = 8
MAX_DURATION = 8


def generate_strategy(price_csv, save_path, min_profit_threshold=0, use_robust=False, error_margin=0.1,
                      use_variable_duration=False):
    """
    根据预测的实时价格确定充放电策略

    策略逻辑：
    1. 寻找最优的充电开始时间tc和放电开始时间td
    2. 充电持续8个时间点（2小时），放电持续8个时间点（2小时）
    3. 充电开始时间：0 <= tc <= 80
    4. 放电开始时间：td >= tc + 8 且 td <= 88
    5. 目标：最大化收益 = sum(放电时段价格) * 1000 - sum(充电时段价格) * 1000

    可变时长模式（use_variable_duration=True）：
    - 充电时长可在 MIN_DURATION 到 MAX_DURATION 之间变化
    - 放电时长可在 MIN_DURATION 到 MAX_DURATION 之间变化
    - 允许多轮充放电

    参数：
        min_profit_threshold: 最小收益阈值，只有当(放电收益-充电成本)超过此阈值时才执行充放电
        use_robust: 是否使用稳健优化策略（考虑预测误差）
        error_margin: 预测误差范围，默认 ±10%
        use_variable_duration: 是否使用可变时长策略（v1_dynamic风格）
    """
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])

    # 按天分组处理
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

        if use_variable_duration:
            # v1_dynamic 可变时长策略：遍历所有充电/放电时长组合
            best_profit = 0
            best_plan = None  # (tc, td, charge_dur, discharge_dur)

            for charge_dur in range(MIN_DURATION, MAX_DURATION + 1):
                for discharge_dur in range(MIN_DURATION, MAX_DURATION + 1):
                    for tc in range(0, 97 - charge_dur):
                        charge_prices = prices[tc:tc + charge_dur]
                        charge_cost = np.sum(charge_prices) * 1000

                        for td in range(tc + charge_dur, 97 - discharge_dur):
                            discharge_prices = prices[td:td + discharge_dur]
                            discharge_revenue = np.sum(discharge_prices) * 1000
                            profit = discharge_revenue - charge_cost

                            if profit > best_profit:
                                best_profit = profit
                                best_plan = (tc, td, charge_dur, discharge_dur)

            # 生成策略
            power = np.zeros(96)
            if best_plan is not None and best_profit > min_profit_threshold:
                tc, td, charge_dur, discharge_dur = best_plan
                power[tc:tc + charge_dur] = -1000
                power[td:td + discharge_dur] = 1000
                total_profit += best_profit
                logger.info(f"日期: {date}, 充电开始: {tc:2d}(+{charge_dur}), 放电开始: {td:2d}(+{discharge_dur}), 收益: {best_profit:10.2f}")
            else:
                skip_count += 1
                logger.info(f"日期: {date}, 跳过交易（收益: {best_profit:.2f} <= 阈值: {min_profit_threshold:.2f}）")

            # 记录 results（variable duration 已在上面处理完）
            for i, (t, p, pr) in enumerate(zip(times, power, prices)):
                results.append({'times': t, '实时价格': pr, 'power': p})
            continue  # 跳过后面的固定8时段逻辑

        elif use_robust:
            # 稳健优化：考虑预测误差的 worst-case
            best_profit_robust = float('-inf')
            best_tc_robust = -1
            best_td_robust = -1

            for tc in range(0, 81):
                for td in range(tc + 8, 89):
                    charge_idx = list(range(tc, tc + 8))
                    discharge_idx = list(range(td, td + 8))

                    # 原始收益
                    charge_cost = np.sum(prices[charge_idx]) * 1000
                    discharge_revenue = np.sum(prices[discharge_idx]) * 1000
                    profit = discharge_revenue - charge_cost

                    # 最坏情况：充电时段价格高估10%，放电时段价格低估10%
                    worst_charge_cost = np.sum(prices[charge_idx] * 1.1) * 1000
                    worst_discharge_revenue = np.sum(prices[discharge_idx] * 0.9) * 1000
                    worst_profit = worst_discharge_revenue - worst_charge_cost

                    # 选最坏情况下最优的方案
                    if worst_profit > best_profit_robust:
                        best_profit_robust = worst_profit
                        best_tc_robust = tc
                        best_td_robust = td

            best_tc = best_tc_robust
            best_td = best_td_robust
            best_profit = best_profit_robust
        else:
            # 原始贪婪策略
            best_profit = 0
            best_tc = -1
            best_td = -1

            for tc in range(0, 81):
                for td in range(tc + 8, 89):
                    charge_prices = prices[tc:tc+8]
                    charge_cost = np.sum(charge_prices) * 1000
                    discharge_prices = prices[td:td+8]
                    discharge_revenue = np.sum(discharge_prices) * 1000
                    profit = discharge_revenue - charge_cost

                    if profit > best_profit:
                        best_profit = profit
                        best_tc = tc
                        best_td = td

        # 生成充放电策略 - 加入阈值兜底
        power = np.zeros(96)
        if best_tc >= 0 and best_td >= 0 and best_profit > min_profit_threshold:
            power[best_tc:best_tc+8] = -1000  # 充电
            power[best_td:best_td+8] = 1000   # 放电
            total_profit += best_profit
            logger.info(f"日期: {date}, 充电开始: {best_tc:2d}, 放电开始: {best_td:2d}, 预期收益: {best_profit:10.2f}")
        else:
            skip_count += 1
            logger.info(f"日期: {date}, 跳过交易（收益: {best_profit:.2f} <= 阈值: {min_profit_threshold:.2f}）")

        for i, (t, p, pr) in enumerate(zip(times, power, prices)):
            results.append({
                'times': t,
                '实时价格': pr,
                'power': p
            })

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
    # 检查数据文件是否存在
    if not os.path.exists(train_feature_path):
        logger.error(f"训练特征文件不存在: {train_feature_path}")
        logger.error("请从赛事官网下载数据并放置到正确的位置")
        logger.info("数据目录结构应为: data/train/mengxi_boundary_anon_filtered.csv, data/train/mengxi_node_price_selected.csv, data/test/test_in_feature_ori.csv")
        exit(1)

    if not os.path.exists(train_label_path):
        logger.error(f"训练标签文件不存在: {train_label_path}")
        exit(1)

    if not os.path.exists(test_feature_path):
        logger.error(f"测试特征文件不存在: {test_feature_path}")
        exit(1)

    logger.info("=" * 60)
    logger.info("开始训练 Sklearn GradientBoosting 模型...")
    logger.info("=" * 60)

    # ==================== 1. 数据准备 ====================
    logger.info("[1/4] 加载数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)
    logger.info(f"  训练特征: {df_feat.shape}")
    logger.info(f"  训练标签: {df_label.shape}")

    # 按 times 内连接对齐
    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    logger.info(f"  合并后: {df_train.shape}")

    df_train = add_time_features(df_train)
    df_train = add_interaction_features(df_train)

    # 方向2: 添加历史电价的滞后特征 (price_lag_96 = 1天前同一时刻)
    # 注意：测试集没有真实电价，滞后特征仅用于训练，不参与预测
    LAG_INTERVALS = 96  # 1天 = 24小时 * 4间隔/小时 = 96个15分钟时段
    logger.info(f"  添加滞后特征 (lag={LAG_INTERVALS})...")
    df_train['price_lag_96'] = df_train[target_col].shift(LAG_INTERVALS)
    df_train['price_lag_192'] = df_train[target_col].shift(192)  # 2天前
    df_train['price_lag_672'] = df_train[target_col].shift(672)  # 1周前

    # 方向3: 加载气象数据(NWP特征)
    nwp_dir = os.path.join(data_dir, 'all_nc')  # 气象数据在 all_nc 目录
    if os.path.exists(nwp_dir) and HAS_NETCDF:
        logger.info("  加载气象数据(NWP特征)...")
        df_nwp = load_nwp_features(nwp_dir, df_train['times'])
        if df_nwp is not None:
            # 对齐气象数据到训练数据时间索引
            df_train = pd.concat([df_train.reset_index(drop=True),
                                  df_nwp.reset_index(drop=True).reindex(df_train.index)], axis=1)
            logger.info(f"  气象特征合并后: {df_train.shape}")

    # 特征列配置 - 分开基础特征（用于训练+预测）和训练专属特征（仅用于训练）
    base_features = feature_cols + ['dayofweek', 'month_sin', 'month_cos', '风光比', '负荷风光差', '水电占比']  # month 用周期性编码避免分布偏移
    lag_features = ['price_lag_96']  # 只保留1天滞后，更稳定
    nwp_features = []  # 气象特征，将在加载NWP后确定

    # 基础特征用于训练和测试
    all_features = base_features.copy()
    # 滞后特征仅用于训练（测试集没有真实电价）
    for lag_feat in lag_features:
        if lag_feat in df_train.columns:
            all_features.append(lag_feat)

    # 过滤掉不存在的列
    all_features = [f for f in all_features if f in df_train.columns]
    logger.info(f"  初始训练特征: {all_features}")

    # 去除滞后特征产生的缺失值（前面的行因移位而缺失）
    lag_cols_in_features = [f for f in all_features if f.startswith('price_lag')]
    df_train_clean = df_train.dropna(subset=[target_col] + lag_cols_in_features).copy()

    # 获取气象特征列名（从df_train_clean中找以'nwp_'开头的列）
    nwp_cols = [col for col in df_train_clean.columns if col.startswith('nwp_')]
    # 基于相关性分析：ch5(-0.44), ch1(+0.30), ch0(-0.29), ch6(+0.23), ch3(+0.19), ch2(-0.19) 都有意义
    # ch4(0.01) 接近0，可以跳过
    important_nwp_cols = [col for col in nwp_cols if col in ['nwp_ch0', 'nwp_ch1', 'nwp_ch2', 'nwp_ch3', 'nwp_ch5', 'nwp_ch6']]
    if important_nwp_cols:
        logger.info(f"  检测到气象特征: {important_nwp_cols}")
        all_features.extend(important_nwp_cols)
        nwp_features = important_nwp_cols

    # 过滤掉不存在的列
    all_features = [f for f in all_features if f in df_train_clean.columns]
    logger.info(f"  最终训练特征: {all_features}")

    logger.info(f"  清理后数据: {df_train_clean.shape}")

    # 训练时使用所有特征（包括滞后）
    X = df_train_clean[all_features].values
    y = df_train_clean[target_col].values

    # 按时间顺序划分，最后20%做验证
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    logger.info(f"  训练集（含滞后特征）: {X_train.shape}, 验证集: {X_val.shape}")

    # ==================== 2. 模型训练（使用滞后特征） ====================
    logger.info("[2/4] 训练模型（含滞后特征）...")

    model = GradientBoostingRegressor(
        n_estimators=300,  # 增加树数量提高拟合能力
        learning_rate=0.05,
        max_depth=5,  # 减少深度降低过拟合
        subsample=0.8,
        verbose=1
    )
    model.fit(X_train, y_train)

    # 验证集评估
    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    logger.info(f"  验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # ==================== 3. 测试集推理（递推预测） ====================
    logger.info("[3/4] 测试集推理（使用递推预测）...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_time_features(df_test)
    df_test = add_interaction_features(df_test)

    # 加载测试集的气象数据
    if os.path.exists(nwp_dir) and HAS_NETCDF:
        logger.info("  加载测试集气象数据(NWP特征)...")
        df_nwp_test = load_nwp_features(nwp_dir, df_test['times'])
        if df_nwp_test is not None:
            # 将 NWP 特征合并到测试数据
            for col in df_nwp_test.columns:
                if col in nwp_features:  # 只添加我们需要的channel
                    df_test[col] = df_nwp_test[col].values
            logger.info(f"  测试集气象特征合并后: {df_test.shape}")

    # 过滤测试集中不存在的特征
    test_features_available = [f for f in base_features if f in df_test.columns]
    logger.info(f"  测试集基础特征: {test_features_available}")

    # 使用递推预测（利用滞后特征）
    y_test_pred = recursive_predict_with_lag(
        model=model,
        df_train=df_train_clean,
        df_test=df_test,
        base_features=test_features_available,
        lag_features=[f for f in lag_features if f in df_train_clean.columns],
        nwp_features=nwp_features,
        target_col=target_col
    )

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)
    logger.info(f"  推理结果已保存: {output_price_path}")
    logger.info(f"  预测天数: {len(df_out) // 96} 天")

    # ==================== 预测结果诊断 ====================
    logger.info("=" * 60)
    logger.info("【预测结果诊断】")
    logger.info(f"  预测价格: min={y_test_pred.min():.4f}, max={y_test_pred.max():.4f}, mean={y_test_pred.mean():.4f}, std={y_test_pred.std():.4f}")
    logger.info(f"  训练价格: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")

    # 训练/测试特征分布对比
    logger.info("【训练/测试特征分布对比】")
    for feat in base_features:
        if feat in df_train_clean.columns and feat in df_test.columns:
            train_mean = df_train_clean[feat].mean()
            test_mean = df_test[feat].mean()
            ratio = test_mean / train_mean if train_mean != 0 else float('inf')
            logger.info(f"  {feat:20s}: train={train_mean:.4f}, test={test_mean:.4f}, ratio={ratio:.4f}")

    # ==================== 4. 生成充放电策略 ====================
    logger.info("[4/4] 生成充放电策略...")

    # ==================== 特征重要性分析 ====================
    logger.info("=" * 60)
    logger.info("【特征重要性分析】")
    importance = model.feature_importances_
    sorted_features = sorted(zip(all_features, importance), key=lambda x: -x[1])
    for feat, imp in sorted_features:
        bar = "█" * int(imp * 100)
        logger.info(f"  {feat:20s}: {imp:.4f} {bar}")

    # 检查低贡献特征
    low_importance = [f for f, i in sorted_features if i < 0.01]
    if low_importance:
        logger.warning(f"  低贡献特征(<0.01): {low_importance} - 建议删除")

    # ==================== 特征共线性分析 ====================
    logger.info("=" * 60)
    logger.info("【特征共线性分析】")
    df_corr = df_train_clean[base_features].corr()
    logger.info("  高度相关的特征对 (|r| > 0.9):")
    found_high_corr = False
    for i in range(len(base_features)):
        for j in range(i+1, len(base_features)):
            corr_val = df_corr.iloc[i, j]
            if abs(corr_val) > 0.9:
                found_high_corr = True
                logger.info(f"    {base_features[i]} <-> {base_features[j]}: {corr_val:.4f}")
    if not found_high_corr:
        logger.info("    无高度相关特征对")

    # 显示中等相关特征
    logger.info("  中等相关特征对 (0.7 < |r| < 0.9):")
    found_med_corr = False
    for i in range(len(base_features)):
        for j in range(i+1, len(base_features)):
            corr_val = df_corr.iloc[i, j]
            if 0.7 < abs(corr_val) <= 0.9:
                found_med_corr = True
                logger.info(f"    {base_features[i]} <-> {base_features[j]}: {corr_val:.4f}")
    if not found_med_corr:
        logger.info("    无中等相关特征对")
    logger.info("=" * 60)

    # 可通过调整 min_profit_threshold 参数来控制策略阈值兜底
    # 例如：min_profit_threshold=1000 表示只有预期收益超过1000时才执行充放电
    # use_variable_duration=True 启用 v1_dynamic 可变时长策略（4-12时段）
    generate_strategy(output_price_path, output_power_path, min_profit_threshold=500, use_robust=False, use_variable_duration=True)
    
    logger.info("=" * 60)
    logger.info("Baseline运行完成！")
    logger.info(f"提交文件: {output_power_path}")
    logger.info("=" * 60)
