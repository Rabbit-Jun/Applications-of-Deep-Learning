import torch
import torch.nn as nn
from train import CrossfitNet, load_processed_data
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, save_path):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # 혼동 행렬 히트맵 생성
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Error'],
                yticklabels=['Normal', 'Error'])
    
    # 제목과 레이블 설정
    plt.title('Confusion Matrix', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 비율 계산
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    # 텍스트 정보 추가
    plt.figtext(0.02, 0.02, f"""
    True Negatives (Normal correctly identified): {tn} ({tn/total*100:.2f}%)
    False Positives (Normal incorrectly identified as Error): {fp} ({fp/total*100:.2f}%)
    False Negatives (Error incorrectly identified as Normal): {fn} ({fn/total*100:.2f}%)
    True Positives (Error correctly identified): {tp} ({tp/total*100:.2f}%)
    """, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # 여백 조정
    plt.tight_layout()
    
    # 이미지 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_model():
    # CUDA 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 저장된 데이터 로드
    print("Loading data...")
    train_x, train_y, test_x, test_y, val_x, val_y = load_processed_data()
    
    # 모델 초기화
    input_size = test_x.shape[1]
    model = CrossfitNet(input_size).to(device)
    
    # 저장된 최적 가중치 로드
    print("Loading best model weights...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # 테스트 데이터를 텐서로 변환
    test_tensor_x = torch.FloatTensor(test_x).to(device)
    test_tensor_y = torch.LongTensor(test_y).to(device)
    
    # 예측 수행
    print("\nMaking predictions...")
    with torch.no_grad():
        outputs = model(test_tensor_x)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # CPU로 데이터 이동
    predictions = predicted.cpu().numpy()
    true_labels = test_tensor_y.cpu().numpy()
    prob_scores = probabilities[:, 1].cpu().numpy()

    # 평가 지표 계산
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc_roc = roc_auc_score(true_labels, prob_scores)

    # 결과 출력
    print("\n=== Test Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # 혼동 행렬 생성 및 저장
    os.makedirs('./analysis_results', exist_ok=True)
    plot_confusion_matrix(
        true_labels,
        predictions,
        './analysis_results/test_confusion_matrix.png'
    )

    # 혼동 행렬 값 출력
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    print("\n=== Confusion Matrix Details ===")
    print(f"True Negatives (Normal correctly identified): {tn} ({tn/total*100:.2f}%)")
    print(f"False Positives (Normal incorrectly identified as Error): {fp} ({fp/total*100:.2f}%)")
    print(f"False Negatives (Error incorrectly identified as Normal): {fn} ({fn/total*100:.2f}%)")
    print(f"True Positives (Error correctly identified): {tp} ({tp/total*100:.2f}%)")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }

if __name__ == "__main__":
    test_model() 