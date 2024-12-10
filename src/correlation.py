import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataload import load_data, load_cached_max_length
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_decomposition import CCA
import scipy.stats as stats

def save_processed_data(train_x, train_y, test_x, test_y, val_x, val_y, save_dir='./processed_data'):
    """처리된 데이터를 numpy 형식으로 저장"""
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/train_x.npy', train_x)
    np.save(f'{save_dir}/train_y.npy', train_y)
    np.save(f'{save_dir}/test_x.npy', test_x)
    np.save(f'{save_dir}/test_y.npy', test_y)
    np.save(f'{save_dir}/val_x.npy', val_x)
    np.save(f'{save_dir}/val_y.npy', val_y)

def load_processed_data(save_dir='./processed_data'):
    """저장된 numpy 데이터 로드"""
    train_x = np.load(f'{save_dir}/train_x.npy')
    train_y = np.load(f'{save_dir}/train_y.npy')
    test_x = np.load(f'{save_dir}/test_x.npy')
    test_y = np.load(f'{save_dir}/test_y.npy')
    val_x = np.load(f'{save_dir}/val_x.npy')
    val_y = np.load(f'{save_dir}/val_y.npy')
    return train_x, train_y, test_x, test_y, val_x, val_y

def analyze_correlations(data, labels, save_dir='./analysis_results'):
    """데이터 상관관계 분석 및 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    print("\n=== 데이터 분석 시작 ===")

    # 1. 데이터 전처리
    print("\n1. 데이터 전처리")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 2. 주요 특성 선택 (f-test 기반)
    print("\n2. 특성 중요도 분석")
    k = 100  # 상위 100개 특성 선택
    selector = SelectKBest(score_func=f_classif, k=k)
    data_selected = selector.fit_transform(data_scaled, labels)
    feature_scores = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(data.shape[1])],
        'Score': selector.scores_,
        'P_value': selector.pvalues_
    })
    
    # 중요 특성 저장
    significant_features = feature_scores[feature_scores['P_value'] < 0.05]
    significant_features = significant_features.sort_values('Score', ascending=False)
    significant_features.to_csv(f'{save_dir}/significant_features.csv', index=False)
    print(f"\n유의미한 특성 수: {len(significant_features)}")
    
    # 3. PCA 분석
    print("\n3. PCA 분석")
    n_components = min(50, data_selected.shape[1])
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_selected)
    
    # 설명된 분산 비율 시각화
    plt.figure(figsize=(10, 6))
    explained_var_ratio = pca.explained_variance_ratio_
    plt.plot(range(1, len(explained_var_ratio) + 1), np.cumsum(explained_var_ratio), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.savefig(f'{save_dir}/pca_explained_variance.png')
    plt.close()
    
    # 4. 상관관계 분석 (선택된 특성들에 대해)
    print("\n4. 상관관계 분석")
    correlation_matrix = np.corrcoef(data_selected.T)
    
    # 상위 특성들의 상관관계 히트맵
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix[:20, :20], 
                cmap='coolwarm', 
                center=0, 
                vmin=-1, 
                vmax=1,
                xticklabels=[f'F{i+1}' for i in range(20)],
                yticklabels=[f'F{i+1}' for i in range(20)])
    plt.title('Top 20 Features Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_heatmap.png')
    plt.close()
    
    # 5. 클래스 별 특성 분포 분석
    print("\n5. 클래스 별 특성 분포 분석")
    top_features = 5
    plt.figure(figsize=(15, 5))
    for i in range(top_features):
        plt.subplot(1, top_features, i+1)
        sns.boxplot(x=labels, y=data_selected[:, i])
        plt.title(f'Feature {i+1}')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_distributions.png')
    plt.close()
    
    # 6. 통계적 분석 결과
    print("\n6. 통계 분석 결과")
    stats_results = []
    for i in range(data_selected.shape[1]):
        normal_data = data_selected[labels == 0, i]
        error_data = data_selected[labels == 1, i]
        
        t_stat, p_val = stats.ttest_ind(normal_data, error_data)
        effect_size = np.abs(np.mean(normal_data) - np.mean(error_data)) / np.std(np.concatenate([normal_data, error_data]))
        
        stats_results.append({
            'feature': f'Feature_{i+1}',
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size
        })
    
    stats_df = pd.DataFrame(stats_results)
    stats_df = stats_df.sort_values('effect_size', ascending=False)
    stats_df.to_csv(f'{save_dir}/statistical_analysis.csv', index=False)
    
    # 결과 요약
    print("\n=== 분석 결과 요약 ===")
    print(f"전체 특성 수: {data.shape[1]}")
    print(f"선택된 중요 특성 수: {k}")
    print(f"유의미한 특성 수 (p < 0.05): {len(significant_features)}")
    print(f"PCA로 설명된 분산 비율: {np.sum(explained_var_ratio):.3f}")
    print("\nTop 5 가장 중요한 특성:")
    print(significant_features.head())
    
    return correlation_matrix, stats_df

def main():
    if os.path.exists('./processed_data/train_x.npy'):
        print("저장된 데이터를 불러옵니다...")
        train_x, train_y, test_x, test_y, val_x, val_y = load_processed_data()
    else:
        print("데이터를 새로 처리하고 저장합니다...")
        cached_max_length = load_cached_max_length()
        train_x, train_y, test_x, test_y, val_x, val_y = load_data(
            './data/train', 
            './data/test', 
            './data/val',
            cached_max_length=cached_max_length
        )
        save_processed_data(train_x, train_y, test_x, test_y, val_x, val_y)

    # 전체 데이터 결합
    X = np.vstack([train_x, test_x, val_x])
    y = np.concatenate([train_y, test_y, val_y])

    print("데이터 분석 시작...")
    print(f"데이터 형태: {X.shape}")
    print(f"클래스 분포: {np.bincount(y)}")

    # 상관관계 분석 수행
    correlation_matrix, stats_results = analyze_correlations(X, y)
    print("\n유의미한 특성 (p-value < 0.05):")
    print(stats_results[stats_results['p_value'] < 0.05])

if __name__ == "__main__":
    main()
