import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import efficientnet_b0
from pathlib import Path
import json

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[1] # Hoặc sửa lại đường dẫn nếu cần
MODELS_DIR = PROJECT_ROOT / "models"
DETECTOR_WEIGHTS = MODELS_DIR / "detector_best.pth"
CLASSIFIER_WEIGHTS = MODELS_DIR / "classifier_best.pth"
LABELS_PATH = MODELS_DIR / "labels.json"

def convert():
    # 1. Load Labels để biết số class
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    num_classes = len(labels)

    print("--- Bắt đầu chuyển đổi ---")

    # 2. Convert DETECTOR
    print("1. Đang load Detector...")
    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2) # 2 class: background + trash
    
    detector.load_state_dict(torch.load(DETECTOR_WEIGHTS, map_location='cpu'))
    detector.eval()

    # Tạo input giả lập để model biết kích thước (cần khớp với ROI size của bạn)
    dummy_input_det = torch.randn(1, 3, 480, 480) 

    print("   -> Đang export Detector sang ONNX (Có thể mất 1-2 phút)...")
    try:
        torch.onnx.export(
            detector, 
            dummy_input_det, 
            MODELS_DIR / "detector.onnx",
            opset_version=11, 
            input_names=['input'], 
            output_names=['boxes', 'labels', 'scores']
        )
        print("   -> Detector OK!")
    except Exception as e:
        print(f"   -> LỖI Export Detector: {e}")

    # 3. Convert CLASSIFIER
    print("2. Đang load Classifier...")
    classifier = efficientnet_b0(weights=None)
    in_features = classifier.classifier[1].in_features
    classifier.classifier[1] = nn.Linear(in_features, num_classes)
    
    classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location='cpu'))
    classifier.eval()

    dummy_input_cls = torch.randn(1, 3, 256, 256)

    print("   -> Đang export Classifier sang ONNX...")
    torch.onnx.export(
        classifier, 
        dummy_input_cls, 
        MODELS_DIR / "classifier.onnx",
        opset_version=11,
        input_names=['input'], 
        output_names=['output']
    )
    print("   -> Classifier OK!")
    print("--- HOÀN TẤT ---")

if __name__ == "__main__":
    convert()