import torch
import torch.nn as nn
from torchvision import models
import warnings
warnings.filterwarnings('ignore')

class VGGClassifier(nn.Module):
    def __init__(self, num_classes=2, input_size=50):
        super(VGGClassifier, self).__init__()
        # VGG16 로드 (weights 파라미터 사용)
        self.vgg = models.vgg16(weights='IMAGENET1K_V1')
        
        # 입력 레이어 수정
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 224 * 224 * 3),
            nn.ReLU(),
            nn.BatchNorm1d(224 * 224 * 3),
            nn.Dropout(0.3)
        )
        
        # 분류기 수정
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = x.view(-1, 3, 224, 224)
        return self.vgg(x)

class ResNeXtClassifier(nn.Module):
    def __init__(self, num_classes=2, input_size=27258):
        super(ResNeXtClassifier, self).__init__()
        # ResNeXt-50 로드 (더 가벼운 버전 사용)
        self.resnext = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 224 * 224 * 3),
            nn.ReLU(),
            nn.BatchNorm1d(224 * 224 * 3),
            nn.Dropout(0.3)
        )
        
        self.resnext.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = x.view(-1, 3, 224, 224)
        return self.resnext(x)

class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=2, input_size=50):
        super(MobileNetClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 224 * 224 * 3),
            nn.ReLU(),
            nn.BatchNorm1d(224 * 224 * 3),
            nn.Dropout(0.3)
        )
        
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = x.view(-1, 3, 224, 224)
        return self.mobilenet(x)

def get_model(model_name, num_classes=2, input_size=27258):
    """모델 선택 함수"""
    models_dict = {
        'vgg': VGGClassifier,
        'resnext': ResNeXtClassifier,
        'mobilenet': MobileNetClassifier
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(models_dict.keys())}")
    
    return models_dict[model_name](num_classes=num_classes, input_size=input_size) 