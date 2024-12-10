import cv2
import mediapipe as mp
import numpy as np
import torch
from train import CrossfitNet

class PosePredictor:
    def __init__(self, model_path='best_model.pth'):
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # CUDA 사용 가능 여부 확인
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 모델 로드
        self.model = CrossfitNet(input_size=27258).to(self.device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        print("Model loaded successfully")

    def extract_keypoints_from_video(self, video_path):
        """비디오에서 모든 프레임의 키포인트 추출"""
        print("Extracting keypoints from video...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None

        all_landmarks = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGR을 RGB로 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 처리
            results = self.pose.process(image)
            
            if results.pose_landmarks:
                # 모든 랜드마크의 x, y, z, visibility 추출
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                all_landmarks.extend(frame_landmarks)
            
            frame_count += 1
            if frame_count % 10 == 0:  # 진행상황 출력
                print(f"Processing frames: {frame_count}/{total_frames}")

        cap.release()
        
        if not all_landmarks:
            print("No landmarks detected in the video")
            return None

        # 입력 크기를 27258로 맞추기 위해 패딩 또는 자르기
        if len(all_landmarks) > 27258:
            all_landmarks = all_landmarks[:27258]
        else:
            all_landmarks.extend([0] * (27258 - len(all_landmarks)))

        return np.array(all_landmarks)

    def predict_video(self, video_path):
        """비디오 전체에 대한 예측"""
        # 비디오에서 키포인트 추출
        features = self.extract_keypoints_from_video(video_path)
        
        if features is None:
            print("Failed to extract features from video")
            return
        
        # 예측
        print("\nMaking prediction...")
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            confidence = probabilities[0][predicted].item()
        
        # 결과 출력
        label = "Error" if predicted.item() == 1 else "Normal"
        print(f"\nPrediction Result:")
        print(f"Label: {label}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    predictor = PosePredictor()
    predictor.predict_video("n.avi")