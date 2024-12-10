from dataload import load_data, load_cached_max_length
import numpy as np

def test_dataload():
    print("\n=== 데이터 로드 테스트 시작 ===")
    
    # 1. cached_max_length 확인
    print("\n1. cached_max_length 확인")
    cached_length = load_cached_max_length()
    print(f"Cached max length: {cached_length}")
    
    # 2. 데이터 로드 시도
    print("\n2. 데이터 로드 시작")
    try:
        train_x, train_y, test_x, test_y, val_x, val_y = load_data(
            './data/train',
            './data/test',
            './data/val',
            cached_max_length=cached_length
        )
        
        # 3. 로드된 데이터 정보 출력
        print("\n3. 로드된 데이터 정보:")
        print(f"Train data shape: {train_x.shape}")
        print(f"Train labels shape: {train_y.shape}")
        print(f"Test data shape: {test_x.shape}")
        print(f"Test labels shape: {test_y.shape}")
        print(f"Val data shape: {val_x.shape}")
        print(f"Val labels shape: {val_y.shape}")
        
        # 4. 레이블 분포 확인
        print("\n4. 레이블 분포:")
        print(f"Train labels: {np.bincount(train_y)}")
        print(f"Test labels: {np.bincount(test_y)}")
        print(f"Val labels: {np.bincount(val_y)}")
        
        print("\n=== 테스트 완료: 성공 ===")
        return True
        
    except Exception as e:
        print(f"\n=== 테스트 실패 ===")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_dataload() 