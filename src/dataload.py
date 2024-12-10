import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, directory, max_length=None, scaler=None, label_encoder=None):
        self.data = []
        self.labels = []
        self.max_length = max_length
        self.scaler = scaler
        self.label_encoder = label_encoder
        print(f"\n{directory} 데이터 로딩 시작...")
        self.load_files(directory)

    def load_files(self, directory):
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        total_files = len(files)
        print(f"총 {total_files}개 파일 처리 예정")

        # 각 데이터셋을 max_length에 맞게 패딩 처리
        for idx, filename in enumerate(files, 1):
            if idx % 100 == 0:  # 100개 파일마다 진행상황 출력
                print(f"Progress: {idx}/{total_files} files processed ({(idx/total_files)*100:.1f}%)")
            
            try:
                df = pd.read_csv(os.path.join(directory, filename), dtype=np.float32)
                file_data = df.values.flatten()
                
                # 0으로 패딩 처리
                if len(file_data) < self.max_length:
                    padded_data = np.pad(file_data, (0, self.max_length - len(file_data)), 
                                       mode='constant', constant_values=0)
                else:
                    padded_data = file_data[:self.max_length]
                
                self.data.append(padded_data)
                label = filename.split('@')[0]
                label = 'normal' if label == 'normal' else 'error'
                self.labels.append(label)
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

        print(f"데이터 로딩 완료: {len(self.data)} 샘플")
        
        try:
            self.data = np.vstack(self.data)
            print(f"데이터 shape: {self.data.shape}")

            # 레이블 인코딩
            if self.label_encoder:
                self.labels = self.label_encoder.transform(self.labels)
                print("레이블 인코딩 완료")
                unique_labels = set(self.labels)
                print(f"Unique labels in dataset: {unique_labels}")
            else:
                self.labels = np.array(self.labels)

            # 스케일링 처리
            if self.scaler:
                print("데이터 스케일링 시작...")
                self.data = self.scaler.transform(self.data)
                print("데이터 스케일링 완료")
        except Exception as e:
            print(f"Error in final processing: {str(e)}")
            print(f"Current data type: {type(self.data)}")
            print(f"Current data length: {len(self.data)}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(int(self.labels[idx]), dtype=torch.long)

    
    
def load_data(train_dir, test_dir, val_dir, cached_max_length=None):
    print("\n=== Data Loading Process Started ===")
    
    # max_length 계산 또는 로드
    if cached_max_length is None:
        print("\nCalculating max_length...")
        max_length = 0
        for directory in [train_dir, test_dir, val_dir]:
            print(f"\nProcessing directory: {directory}")
            files = [f for f in os.listdir(directory) if f.endswith('.csv')]
            total_files = len(files)
            
            for idx, filename in enumerate(files, 1):
                if idx % 100 == 0:  # 100개 파일마다 진행상황 출력
                    print(f"Progress: {idx}/{total_files} files checked")
                df = pd.read_csv(os.path.join(directory, filename), dtype=np.float32)
                max_length = max(max_length, df.values.flatten().shape[0])
        
        print(f"\nFinal max_length calculated: {max_length}")
        np.save('./cached_max_length.npy', max_length)
    else:
        max_length = cached_max_length
        print(f"\nUsing cached max_length: {max_length}")

    print("\nInitializing preprocessing components...")
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    print("\nLoading training dataset...")
    temp_train_dataset = CustomDataset(train_dir, max_length=max_length)
    print("Fitting label encoder...")
    label_encoder.fit(temp_train_dataset.labels)

    print("\nProcessing training data...")
    train_dataset = CustomDataset(train_dir, max_length=max_length, label_encoder=label_encoder)
    print("Fitting scaler and transforming training data...")
    train_data = scaler.fit_transform(train_dataset.data)
    train_labels = train_dataset.labels

    print("\nProcessing validation data...")
    val_dataset = CustomDataset(val_dir, max_length=max_length, scaler=scaler, label_encoder=label_encoder)
    val_data = val_dataset.data
    val_labels = val_dataset.labels

    print("\nProcessing test data...")
    test_dataset = CustomDataset(test_dir, max_length=max_length, scaler=scaler, label_encoder=label_encoder)
    test_data = test_dataset.data
    test_labels = test_dataset.labels

    print("\n=== Data Loading Process Completed ===")
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    return train_data, train_labels, test_data, test_labels, val_data, val_labels

# max_length를 로드하는 함수
def load_cached_max_length():
    try:
        return np.load('./cached_max_length.npy')
    except:
        return None

