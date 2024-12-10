import cv2
import mediapipe as mp
import numpy as np
import torch
from models import get_model
import argparse

class PosePredictor:
    def __init__(self, model_name='vgg', model_path=None):
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
        self.model = get_model(model_name, input_size=27258).to(self.device)
        if model_path is None:
            model_path = f'best_model_{model_name}.pth'
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        print("Model loaded successfully")

    def process_video(self, input_path):
        """비디오 처리 및 결과 출력"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return
        
        all_features = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 키포인트 추출
            features = self.extract_keypoints(frame)
            if features is not None:
                all_features.append(features)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        
        if not all_features:
            print("No poses detected in the video")
            return
        
        # 모든 프레임의 특성을 하나로 결합
        features = np.mean(all_features, axis=0)
        
        # 예측
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

    def extract_keypoints(self, frame):
        """프레임에서 키포인트 추출"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            padded_features = np.zeros(27258)
            padded_features[:len(landmarks)] = landmarks
            
            return padded_features
        
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg',
                      choices=['vgg', 'resnext', 'mobilenet'],
                      help='Model architecture to use')
    parser.add_argument('--video', type=str, required=True,
                      help='Path to input video file')
    args = parser.parse_args()
    
    predictor = PosePredictor(model_name=args.model)
    predictor.process_video(args.video)