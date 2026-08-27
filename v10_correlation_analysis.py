"""
v10 特征相关性热图分析
检查共线性问题：高度相关的特征对可能让模型学得糊涂
"""
import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')

feature_cols = ['系统负荷预测值', '风光总加预测值', '风电预测值', '光伏预测值',
                '水电预测值', '非市场化机组预测值']
time_features = ['hour', 'month']


def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


if __name__ == '__main__':
    print("=" * 60)
    print("Feature Correlation Heatmap Analysis")
    print("=" * 60)

    # ==================== 数据准备 ====================
    print("\n[1/2] Loading data...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)

    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    df_train = add_time_features(df_train)

    # v9 保留的8个特征
    all_features = feature_cols + time_features
    print(f"  Features: {all_features}")

    # 计算相关性矩阵
    corr_matrix = df_train[all_features].corr()

    # ==================== 打印相关性矩阵 ====================
    print("\n[2/2] Correlation Matrix:")
    print("-" * 80)

    # 格式化打印
    print(f"\n{'Feature':<20}", end="")
    for feat in all_features:
        short = feat[:8]
        print(f"{short:>10}", end="")
    print()

    for i, feat1 in enumerate(all_features):
        short1 = feat1[:18]
        print(f"{short1:<20}", end="")
        for j, feat2 in enumerate(all_features):
            corr_val = corr_matrix.iloc[i, j]
            print(f"{corr_val:>10.4f}", end="")
        print()

    # ==================== 找出高相关特征对 ====================
    print("\n" + "=" * 60)
    print("Highly Correlated Feature Pairs (|r| > 0.7)")
    print("=" * 60)

    high_corr_pairs = []
    for i in range(len(all_features)):
        for j in range(i+1, len(all_features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((all_features[i], all_features[j], corr_val))

    if high_corr_pairs:
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            print(f"  {feat1} <-> {feat2}: {corr:.4f}")
    else:
        print("  No highly correlated pairs found")

    # ==================== 中等相关 ====================
    print("\n" + "=" * 60)
    print("Medium Correlated Feature Pairs (0.5 < |r| < 0.7)")
    print("=" * 60)

    med_corr_pairs = []
    for i in range(len(all_features)):
        for j in range(i+1, len(all_features)):
            corr_val = corr_matrix.iloc[i, j]
            if 0.5 < abs(corr_val) <= 0.7:
                med_corr_pairs.append((all_features[i], all_features[j], corr_val))

    if med_corr_pairs:
        for feat1, feat2, corr in sorted(med_corr_pairs, key=lambda x: -abs(x[2])):
            print(f"  {feat1} <-> {feat2}: {corr:.4f}")
    else:
        print("  No medium correlated pairs found")

    # ==================== 建议 ====================
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)

    if high_corr_pairs:
        print("\n[!] High correlation detected. Consider removing one of:")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"    - {feat1} vs {feat2} (r={corr:.2f})")
        print("\n    Reason: Both features convey similar information,")
        print("            causing multicollinearity. Model may benefit from")
        print("            keeping only one.")
    else:
        print("\n[OK] No severe multicollinearity detected")

    print("\n" + "=" * 60)
