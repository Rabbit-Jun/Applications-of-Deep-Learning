import torch
import numpy as np
from models import get_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from sklearn.decomposition import PCA

def load_test_data():
    """테스트 데이터만 로드하고 PCA 적용"""
    print("Loading test data...")
    try:
        test_x = np.load('processed_data/test_x.npy')
        test_y = np.load('processed_data/test_y.npy')
        
        # PCA 모델 로드 또는 새로 학습
        try:
            pca = np.load('processed_data/pca_model.npy', allow_pickle=True).item()
        except:
            print("Training new PCA model...")
            pca = PCA(n_components=50)
            train_x = np.load('processed_data/train_x.npy')
            pca.fit(train_x)
            np.save('processed_data/pca_model.npy', pca)
        
        # 테스트 데이터에 PCA 적용
        test_x = pca.transform(test_x)
        
        print(f"Test data shape after PCA: {test_x.shape}")
        return test_x, test_y
    except Exception as e:
        print(f"Error loading test data: {e}")
        raise

def test_model(model_name):
    # 결과 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f'test_results_{model_name}_{timestamp}'
    os.makedirs(result_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 테스트 데이터만 로드
    test_x, test_y = load_test_data()
    
    # 모델 생성
    model = get_model(model_name, input_size=50).to(device)
    
    # 모델 가중치 로드
    model_path = f'best_model_{model_name}.pth'
    print(f"Loading best model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    
    # 테스트 모드로 설정
    model.eval()
    
    # 예측 수행
    print("\nPerforming predictions...")
    with torch.no_grad():
        test_x_tensor = torch.FloatTensor(test_x).to(device)
        outputs = model(test_x_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
    
    # 결과 평가
    print("\nGenerating classification report...")
    class_report = classification_report(test_y, predicted)
    print("\nClassification Report:")
    print(class_report)
    
    # 분류 보고서 저장
    with open(f'{result_dir}/classification_report.txt', 'w') as f:
        f.write(class_report)
    
    # 혼동 행렬 생성 및 시각화
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(test_y, predicted)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 정확도 등의 메트릭 추가
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.4f}', 
             ha='center', va='center', transform=plt.gca().transAxes)
    
    # 혼동 행렬 저장
    plt.tight_layout()
    plt.savefig(f'{result_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 예측 확률 분포 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities[:, 1], bins=50, alpha=0.7)
    plt.title(f'Prediction Probability Distribution - {model_name}')
    plt.xlabel('Probability of Positive Class')
    plt.ylabel('Count')
    plt.savefig(f'{result_dir}/probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 상세 결과 저장
    results = {
        'true_labels': test_y,
        'predicted_labels': predicted,
        'probabilities': probabilities
    }
    np.save(f'{result_dir}/detailed_results.npy', results)
    
    print(f"\nAll results have been saved to directory: {result_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg',
                       choices=['vgg', 'wide_resnet', 'mobilenet'])
    args = parser.parse_args()
    
    test_model(args.model)

if __name__ == "__main__":
    main() 