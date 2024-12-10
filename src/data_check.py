import numpy as np
import os

# 현재 작업 디렉토리 확인
print("현재 작업 디렉토리:", os.getcwd()) 
# npy 파일 로드
data = np.load(r'./processed_data/test_x.npy')

# 데이터의 형태 확인
print("데이터 형태:", data.shape)

# 데이터의 첫 몇 개 샘플 확인
print("데이터 샘플:", data[:5])  # 첫 5개 샘플 출력

# 전체 특성 수 확인
num_features = data.shape[1]
print("전체 특성 수:", num_features)
