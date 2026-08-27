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

    nc_files = glob.glob(os.path.join(nc_dir, '*.nc'))
    if not nc_files:
        logger.warning(f"气象数据目录 {nc_dir} 中未找到 .nc 文件")
        return None

    logger.info(f"找到 {len(nc_files)} 个气象数据文件")

    # 用于存储各文件提取的特征
    feature_list = []

    for nc_path in nc_files:
        try:
            dataset = nc.Dataset(nc_path, 'r')

            # 尝试找到风速和辐照度相关变量
            var_names = list(dataset.variables.keys())
            logger.info(f"处理文件: {os.path.basename(nc_path)}, 变量: {var_names}")

            # 常见变量名映射（根据实际数据调整）
            target_vars = []
            for var in var_names:
                var_lower = var.lower()
                if 'u100' in var_lower or 'v100' in var_lower or 'wind' in var_lower:
                    target_vars.append(('wind', var))
                elif 'ghi' in var_lower or 'dswrf' in var_lower or 'irradi' in var_lower or 'solar' in var_lower:
                    target_vars.append(('ghi', var))

            # 提取并空间平均
            for prefix, var_name in target_vars:
                if var_name in dataset.variables:
                    data = dataset.variables[var_name]
                    # 假设维度顺序为 (time, lat, lon) 或类似结构
                    if len(data.shape) >= 3:
                        # 对 lat/lon 做空间平均
                        spatial_avg = np.mean(data, axis=(-2, -1))
                    else:
                        spatial_avg = data

                    # 时间重采样：从每小时转换到 15 分钟
                    n_time = len(spatial_avg)
                    n_target = len(times_index)

                    # 创建时间索引（假设从 0 点开始）
                    nc_times = np.arange(n_time)

                    # 线性插值到 15 分钟粒度
                    target_indices = np.arange(n_target) / intervals_per_hour
                    if n_time > 1:
                        interpolated = np.interp(target_indices, nc_times, spatial_avg)
                    else:
                        interpolated = np.full(n_target, spatial_avg[0]) if n_time == 1 else None

                    if interpolated is not None:
                        col_name = f"{prefix}_spatial_avg"
                        if col_name in [df.columns for df in feature_list if hasattr(df, 'columns')]:
                            # 如果已存在该列，合并
                            pass
                        feature_list.append(pd.DataFrame({col_name: interpolated}, index=times_index))

            dataset.close()
            logger.info(f"已处理: {os.path.basename(nc_path)}")

        except Exception as e:
            logger.error(f"处理文件 {nc_path} 时出错: {e}")
            continue

    if not feature_list:
        logger.warning("未能从气象数据中提取任何特征")
        return None

    # 合并所有特征
    df_nwp = pd.concat(feature_list, axis=1)
    df_nwp.index = times_index
    logger.info(f"气象特征加载完成，形状: {df_nwp.shape}")

    return df_nwp


# ==================== 充放电策略生成 ====================
def generate_strategy(price_csv, save_path, min_profit_threshold=0):
    """
    根据预测的实时价格确定充放电策略

    策略逻辑：
    1. 寻找最优的充电开始时间tc和放电开始时间td
    2. 充电持续8个时间点（2小时），放电持续8个时间点（2小时）
    3. 充电开始时间：0 <= tc <= 80
    4. 放电开始时间：td >= tc + 8 且 td <= 88
    5. 目标：最大化收益 = sum(放电时段价格) * 1000 - sum(充电时段价格) * 1000

    参数：
        min_profit_threshold: 最小收益阈值，只有当(放电收益-充电成本)超过此阈值时才执行充放电
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

        best_profit = 0
        best_tc = -1
        best_td = -1

        # 遍历所有可能的充电和放电开始时间
        for tc in range(0, 81):  # 0 <= tc <= 80
            charge_prices = prices[tc:tc+8]
            charge_cost = np.sum(charge_prices) * 1000  # 充电成本

            for td in range(tc + 8, 89):  # td >= tc + 8 且 td <= 88
                discharge_prices = prices[td:td+8]
                discharge_revenue = np.sum(discharge_prices) * 1000  # 放电收入

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

    # 方向2: 添加历史电价的滞后特征 (price_lag_96 = 1天前同一时刻)
    # 注意：测试集没有真实电价，滞后特征仅用于训练，不参与预测
    LAG_INTERVALS = 96  # 1天 = 24小时 * 4间隔/小时 = 96个15分钟时段
    logger.info(f"  添加滞后特征 (lag={LAG_INTERVALS})...")
    df_train['price_lag_96'] = df_train[target_col].shift(LAG_INTERVALS)
    df_train['price_lag_192'] = df_train[target_col].shift(192)  # 2天前
    df_train['price_lag_672'] = df_train[target_col].shift(672)  # 1周前

    # 方向3: 加载气象数据(NWP特征)
    nwp_dir = os.path.join(data_dir, 'train')
    if os.path.exists(nwp_dir) and HAS_NETCDF:
        logger.info("  加载气象数据(NWP特征)...")
        df_nwp = load_nwp_features(nwp_dir, df_train['times'])
        if df_nwp is not None:
            # 对齐气象数据到训练数据时间索引
            df_train = pd.concat([df_train.reset_index(drop=True),
                                  df_nwp.reset_index(drop=True).reindex(df_train.index)], axis=1)
            logger.info(f"  气象特征合并后: {df_train.shape}")

    # 特征列配置 - 分开基础特征（用于训练+预测）和训练专属特征（仅用于训练）
    base_features = feature_cols + ['hour', 'minute', 'dayofweek', 'month']
    lag_features = ['price_lag_96', 'price_lag_192', 'price_lag_672']

    # 基础特征用于训练和测试
    all_features = base_features.copy()
    # 滞后特征仅用于训练（测试集没有真实电价）
    for lag_feat in lag_features:
        if lag_feat in df_train.columns:
            all_features.append(lag_feat)

    # 过滤掉不存在的列
    all_features = [f for f in all_features if f in df_train.columns]
    logger.info(f"  训练特征: {all_features}")

    # 去除滞后特征产生的缺失值（前面的行因移位而缺失）
    lag_cols_in_features = [f for f in all_features if f.startswith('price_lag')]
    df_train_clean = df_train.dropna(subset=[target_col] + lag_cols_in_features)

    logger.info(f"  清理后数据: {df_train_clean.shape}")

    # 训练时使用所有特征（包括滞后），但预测时只用基础特征
    # 因此需要训练两个模型：model（所有特征，用于分析）和 model_base（基础特征，用于预测）
    X = df_train_clean[all_features].values
    y = df_train_clean[target_col].values

    # 按时间顺序划分，最后20%做验证
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    logger.info(f"  训练集（含滞后特征）: {X_train.shape}, 验证集: {X_val.shape}")

    # ==================== 2. 模型训练（基础特征模型，用于预测） ====================
    logger.info("[2/4] 训练模型（基础特征）...")
    X_base = df_train_clean[base_features].values
    y_base = df_train_clean[target_col].values

    split_idx_base = int(len(X_base) * 0.8)
    X_train_base, X_val_base = X_base[:split_idx_base], X_base[split_idx_base:]
    y_train_base, y_val_base = y_base[:split_idx_base], y_base[split_idx_base:]

    logger.info(f"  基础特征训练集: {X_train_base.shape}, 验证集: {X_val_base.shape}")

    model_base = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        verbose=1
    )
    model_base.fit(X_train_base, y_train_base)

    # 验证集评估（使用基础特征训练的模型，用于最终预测）
    y_val_pred = model_base.predict(X_val_base)
    rmse = np.sqrt(mean_squared_error(y_val_base, y_val_pred))
    mae = mean_absolute_error(y_val_base, y_val_pred)
    logger.info(f"  验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # ==================== 3. 测试集推理 ====================
    logger.info("[3/4] 测试集推理...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_time_features(df_test)

    # 测试集特征：只使用基础特征，不包含滞后特征（测试集没有真实电价）
    test_features = base_features.copy()
    # 过滤测试集中不存在的特征
    test_features = [f for f in test_features if f in df_test.columns]
    logger.info(f"  测试集特征: {test_features}")

    X_test = df_test[test_features].values
    y_test_pred = model_base.predict(X_test)

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)
    logger.info(f"  推理结果已保存: {output_price_path}")
    logger.info(f"  预测天数: {len(df_out) // 96} 天")

    # ==================== 4. 生成充放电策略 ====================
    logger.info("[4/4] 生成充放电策略...")

    # ==================== 特征重要性分析 ====================
    logger.info("=" * 60)
    logger.info("特征重要性分析（从高到低）:")
    importance = model_base.feature_importances_
    sorted_features = sorted(zip(base_features, importance), key=lambda x: -x[1])
    for feat, imp in sorted_features:
        bar = "█" * int(imp * 50)
        logger.info(f"  {feat:20s}: {imp:.4f} {bar}")

    # 检查是否有贡献度接近0的特征
    low_importance = [f for f, i in sorted_features if i < 0.01]
    if low_importance:
        logger.warning(f"低贡献特征(<0.01): {low_importance}")
    logger.info("=" * 60)

    # 可通过调整 min_profit_threshold 参数来控制策略阈值兜底
    # 例如：min_profit_threshold=1000 表示只有预期收益超过1000时才执行充放电
    generate_strategy(output_price_path, output_power_path, min_profit_threshold=0)
    
    logger.info("=" * 60)
    logger.info("Baseline运行完成！")
    logger.info(f"提交文件: {output_power_path}")
    logger.info("=" * 60)
