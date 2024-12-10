import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models import get_model
import wandb
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import gc
from sklearn.decomposition import PCA

def calculate_metrics(y_true, y_pred, y_prob=None):
    """모든 평가 지표 계산"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    
    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, model_name):
    best_val_loss = float('inf')
    total_batches = len(train_loader)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_preds = []
        train_probs = []
        train_labels = []
        
        # 진행 상황 표시
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx % 10 == 0:  # 10배치마다 진행상황 출력
                print(f'Training batch: {batch_idx}/{total_batches} '
                      f'({(batch_idx/total_batches)*100:.1f}%)')
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 예측값과 실제값 저장
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_probs.extend(probs[:, 1].detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        print('\nValidation 시작...')
        # 검증 모드
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                if batch_idx % 5 == 0:  # 5배치마다 진행상황 출력
                    print(f'Validation batch: {batch_idx}/{len(val_loader)} '
                          f'({(batch_idx/len(val_loader))*100:.1f}%)')
                
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 예측값과 실제값 저장
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs[:, 1].detach().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # 평균 손실 계산
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        # 평가 지표 계산
        train_metrics = calculate_metrics(train_labels, train_preds, train_probs)
        val_metrics = calculate_metrics(val_labels, val_preds, val_probs)
        
        print('\n' + '=' * 50)
        print(f'Epoch {epoch+1}/{num_epochs} 결과:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print('\nTrain Metrics:')
        for metric, value in train_metrics.items():
            print(f'  {metric}: {value:.4f}')
        print('\nValidation Metrics:')
        for metric, value in val_metrics.items():
            print(f'  {metric}: {value:.4f}')
        print('=' * 50)
        
        # wandb에 기록
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'val_precision': val_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_recall': val_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_f1': val_metrics['f1'],
            'train_auc_roc': train_metrics['auc_roc'],
            'val_auc_roc': val_metrics['auc_roc']
        })
        
        # 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')
            print(f'\nModel saved with validation loss: {val_loss:.4f}')
        
        # 학습률 조정
        scheduler.step(val_loss)
        
        # 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()

def load_processed_data():
    """넘파이로 저장된 전처리된 데이터 로드 및 PCA 적용"""
    try:
        # 데이터 로드
        train_x = np.load('processed_data/train_x.npy')
        train_y = np.load('processed_data/train_y.npy')
        test_x = np.load('processed_data/test_x.npy')
        test_y = np.load('processed_data/test_y.npy')
        val_x = np.load('processed_data/val_x.npy')
        val_y = np.load('processed_data/val_y.npy')
        
        # PCA 적용 (50개 컴포넌트)
        pca = PCA(n_components=50)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)
        val_x = pca.transform(val_x)
        
        print("Data loaded and PCA applied successfully")
        print(f"Train set shape after PCA: {train_x.shape}")
        print(f"Test set shape after PCA: {test_x.shape}")
        print(f"Validation set shape after PCA: {val_x.shape}")
        print(f"Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
        
        return train_x, train_y, test_x, test_y, val_x, val_y
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg', 
                       choices=['vgg', 'wide_resnet', 'mobilenet'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    args = parser.parse_args()

    # wandb 초기화
    wandb.init(project="crossfit-analysis")
    
    # 전처리된 데이터 로드
    train_x, train_y, test_x, test_y, val_x, val_y = load_processed_data()
    
    # 입력 크기를 27258로 설정
    input_size = train_x.shape[1]
    
    # float32로 변환
    train_x = train_x.astype(np.float32)
    val_x = val_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    
    # 데이터 정규화
    train_mean = train_x.mean(axis=0)
    train_std = train_x.std(axis=0)
    
    train_x = (train_x - train_mean) / (train_std + 1e-7)
    val_x = (val_x - train_mean) / (train_std + 1e-7)
    test_x = (test_x - train_mean) / (train_std + 1e-7)
    
    # 데이터로더 생성
    train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.LongTensor(val_y))
    test_dataset = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 생성 (input_size 전달)
    model = get_model(args.model, input_size=input_size).to(device)
    print(f"Model created with input size: {input_size}")
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # wandb 설정
    wandb.config.update(args.__dict__)

    try:
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs, args.model)
    except Exception as e:
        print(f"Training error: {e}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
