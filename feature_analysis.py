"""
特征分析模块 - 排列重要性 + 相关性热图
功能:
1. 排列重要性 (Permutation Importance)
2. 特征相关性热图
3. 综合报告输出
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 路径配置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
output_dir = os.path.join(current_dir, 'output', 'feature_analysis')
os.makedirs(output_dir, exist_ok=True)

feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'

def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df

def save_validation_data(X_val, y_val, y_val_pred, feature_names):
    """保存验证集数据到npz文件"""
    npz_path = os.path.join(output_dir, 'validation_data.npz')
    np.savez(npz_path,
             X_val=X_val,
             y_val=y_val,
             y_val_pred=y_val_pred)
    print(f"  验证集数据已保存: {npz_path}")
    return npz_path

def compute_permutation_importance(model, X_val, y_val, feature_names):
    """
    计算排列重要性
    原理: 打乱某个特征的数值，看RMSE上升多少。上升越多越重要。
    """
    print("\n[1] 计算排列重要性...")
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=10,   # 重复次数
        random_state=42,
        n_jobs=-1
    )

    # 构建结果DataFrame
    perm_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
        'importance_min': result.importances.min(axis=1),
        'importance_max': result.importances.max(axis=1)
    })
    perm_imp_df = perm_imp_df.sort_values('importance_mean', ascending=False)

    # 保存CSV
    csv_path = os.path.join(output_dir, 'permutation_importance.csv')
    perm_imp_df.to_csv(csv_path, index=False)
    print(f"  排列重要性已保存: {csv_path}")

    # 打印Top5
    print("\n  排列重要性 Top5:")
    for i, row in perm_imp_df.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance_mean']:.6f} (±{row['importance_std']:.6f})")

    return perm_imp_df

def compute_correlation_heatmap(X_train, feature_names):
    """
    计算特征相关性热图
    目的: 检测共线性问题 - 高度相关的特征让模型学得糊涂
    """
    print("\n[2] 计算特征相关性热图...")

    # 创建DataFrame
    X_df = pd.DataFrame(X_train, columns=feature_names)

    # 计算相关性矩阵
    corr_matrix = X_df.corr()

    # 保存相关性矩阵CSV
    corr_csv_path = os.path.join(output_dir, 'correlation_matrix.csv')
    corr_matrix.to_csv(corr_csv_path)
    print(f"  相关性矩阵已保存: {corr_csv_path}")

    # 绘制热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                 center=0, square=True, linewidths=0.5,
                 cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Heatmap\n特征相关性热图', fontsize=14)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  相关性热图已保存: {heatmap_path}")

    # 分析共线性问题
    print("\n  高相关性特征对 (|r| > 0.8):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                high_corr_pairs.append((feat1, feat2, corr_val))
                print(f"    {feat1} <-> {feat2}: {corr_val:.3f}")

    if not high_corr_pairs:
        print("    未发现高度相关特征对")

    return corr_matrix, high_corr_pairs

def compute_feature_importance_ranking(model, feature_names, perm_imp_df):
    """
    综合特征重要性排名
    结合模型内置重要性 + 排列重要性
    """
    print("\n[3] 生成综合特征重要性排名...")

    # 模型内置重要性
    built_in_imp = pd.DataFrame({
        'feature': feature_names,
        'built_in_importance': model.feature_importances_
    }).sort_values('built_in_importance', ascending=False)

    # 合并排列重要性
    ranking_df = built_in_imp.merge(
        perm_imp_df[['feature', 'importance_mean']],
        on='feature'
    )
    ranking_df = ranking_df.rename(columns={'importance_mean': 'permutation_importance'})
    ranking_df = ranking_df.sort_values('permutation_importance', ascending=False)
    ranking_df['rank'] = range(1, len(ranking_df) + 1)

    # 保存
    ranking_path = os.path.join(output_dir, 'feature_ranking.csv')
    ranking_df.to_csv(ranking_path, index=False)
    print(f"  综合排名已保存: {ranking_path}")

    print("\n  综合排名:")
    for _, row in ranking_df.iterrows():
        print(f"    #{int(row['rank'])} {row['feature']}: "
              f"内置={row['built_in_importance']:.4f}, "
              f"排列={row['permutation_importance']:.4f}")

    return ranking_df

def generate_analysis_report(corr_matrix, high_corr_pairs, perm_imp_df, ranking_df):
    """生成文本分析报告"""
    report_path = os.path.join(output_dir, 'analysis_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("特征分析报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("## 1. 排列重要性 (Permutation Importance)\n")
        f.write("原理: 打乱某个特征的数值，看RMSE上升多少。上升越多越重要。\n")
        f.write("比feature_importances_更可靠，因为反映了特征的真实贡献。\n\n")
        f.write("Top5 重要特征:\n")
        for i, row in perm_imp_df.head(5).iterrows():
            f.write(f"  - {row['feature']}: {row['importance_mean']:.6f}\n")

        f.write("\n## 2. 特征相关性分析\n")
        f.write("目的: 检测共线性问题 - 高度相关的特征让模型学得糊涂\n\n")

        if high_corr_pairs:
            f.write("高相关性特征对 (|r| > 0.8):\n")
            for feat1, feat2, corr_val in high_corr_pairs:
                f.write(f"  - {feat1} <-> {feat2}: r={corr_val:.3f}\n")
            f.write("\n建议: 考虑移除其中一个高度相关的特征\n")
        else:
            f.write("未发现高度相关特征对 (|r| > 0.8)\n")

        f.write("\n## 3. 综合建议\n")
        f.write("- 优先保留排列重要性高的特征\n")
        f.write("- 对于高度相关特征对，可根据业务理解保留其一\n")

    print(f"\n  分析报告已保存: {report_path}")
    return report_path

def main():
    print("=" * 60)
    print("特征分析模块 - 排列重要性 + 相关性热图")
    print("=" * 60)

    # 1. 加载数据
    print("\n[加载数据]")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)
    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    df_train = add_time_features(df_train)

    all_features = feature_cols + ['hour', 'minute', 'dayofweek', 'month']
    print(f"  特征数量: {len(all_features)}")
    print(f"  样本数量: {len(df_train)}")

    X = df_train[all_features].values
    y = df_train[target_col].values

    # 2. 划分训练/验证集
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # 3. 训练模型 (用于特征重要性分析)
    print("\n[训练模型]")
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, verbose=0
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"  验证集 RMSE: {rmse:.6f}")

    # 4. 保存验证集数据
    save_validation_data(X_val, y_val, y_val_pred, all_features)

    # 5. 计算排列重要性
    perm_imp_df = compute_permutation_importance(model, X_val, y_val, all_features)

    # 6. 计算相关性热图
    corr_matrix, high_corr_pairs = compute_correlation_heatmap(X_train, all_features)

    # 7. 综合排名
    ranking_df = compute_feature_importance_ranking(model, all_features, perm_imp_df)

    # 8. 生成分析报告
    generate_analysis_report(corr_matrix, high_corr_pairs, perm_imp_df, ranking_df)

    print("\n" + "=" * 60)
    print("特征分析完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 列出生成的文件
    print("\n生成的文件:")
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        print(f"  - {f} ({size/1024:.1f} KB)")

if __name__ == '__main__':
    main()