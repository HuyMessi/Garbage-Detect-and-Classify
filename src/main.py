import time
import json
import threading
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from openvino.runtime import Core

# ----------------------------
# CẤU HÌNH
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DETECTOR_PTH = MODELS_DIR / "detector_best.pth"
CLASSIFIER_ONNX = MODELS_DIR / "classifier.onnx"
LABELS_PATH = MODELS_DIR / "labels.json"

CAM_INDEX = 0
ROI_SIZE = 480
CONF_THRESH = 0.5

# ----------------------------
# CLASS XỬ LÝ AI CHẠY NGẦM
# ----------------------------
class AIThread(threading.Thread):
    def __init__(self, detector, classifier_req, labels, output_layer):
        super().__init__()
        self.detector = detector
        self.infer_request = classifier_req
        self.labels = labels
        self.output_layer = output_layer
        
        self.frame_to_process = None
        self.result_box = None
        self.result_label = ""
        self.result_conf = 0.0
        
        self.running = True
        self.has_new_frame = False
        self.lock = threading.Lock() # Khóa để tránh tranh chấp dữ liệu

    def update_frame(self, frame):
        """Nhận frame mới từ luồng chính"""
        with self.lock:
            self.frame_to_process = frame.copy()
            self.has_new_frame = True

    def get_result(self):
        """Trả kết quả mới nhất cho luồng chính"""
        with self.lock:
            return self.result_box, self.result_label, self.result_conf

    def stop(self):
        self.running = False

    def run(self):
        """Vòng lặp vĩnh cửu của AI (Chạy song song)"""
        while self.running:
            if not self.has_new_frame:
                time.sleep(0.01) # Nghỉ tí nếu không có việc
                continue

            # Lấy frame ra để xử lý
            with self.lock:
                roi_frame = self.frame_to_process
                self.has_new_frame = False
            
            # --- 1. DETECT (CPU) ---
            # (Code detect cũ của bạn)
            device = next(self.detector.parameters()).device
            roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(roi_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                out = self.detector(img_tensor)[0]
            
            boxes = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            
            valid_idx = np.where(scores > CONF_THRESH)[0]
            best_idx = -1
            max_area = 0
            
            for i in valid_idx:
                b = boxes[i]
                area = (b[2]-b[0]) * (b[3]-b[1])
                if area > max_area:
                    max_area = area
                    best_idx = i
            
            # --- 2. CLASSIFY (GPU) ---
            new_box = None
            new_label = ""
            new_conf = 0.0

            if best_idx != -1:
                new_box = boxes[best_idx]
                bx1, by1, bx2, by2 = new_box.astype(int)
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(ROI_SIZE, bx2), min(ROI_SIZE, by2)
                
                object_img = roi_frame[by1:by2, bx1:bx2]
                
                if object_img.size > 0:
                    # Preprocess for OpenVINO
                    img_resized = cv2.resize(object_img, (256, 256))
                    img_input = img_resized.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    img_input = (img_input - mean) / std
                    img_input = img_input.transpose(2, 0, 1)[None, ...] # HWC->NCHW

                    # Inference
                    res = self.infer_request.infer({0: img_input})
                    logits = res[self.output_layer][0]
                    probs = np.exp(logits) / np.sum(np.exp(logits))
                    cls_id = np.argmax(probs)
                    
                    new_label = self.labels[cls_id]
                    new_conf = probs[cls_id]
            
            # Cập nhật kết quả vào biến chung
            with self.lock:
                self.result_box = new_box
                self.result_label = new_label
                self.result_conf = new_conf

# ----------------------------
# MAIN PROGRAM
# ----------------------------
def main():
    # 1. SETUP MODELS
    print("Loading Models...")
    with open(LABELS_PATH, "r") as f: labels = json.load(f)

    # PyTorch Detector (CPU)
    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    detector.load_state_dict(torch.load(DETECTOR_PTH, map_location="cpu"))
    detector.eval()

    # OpenVINO Classifier (GPU)
    ie = Core()
    model_onnx = ie.read_model(model=CLASSIFIER_ONNX)
    compiled_model = ie.compile_model(model=model_onnx, device_name="GPU")
    infer_req = compiled_model.create_infer_request()
    out_layer = compiled_model.output(0)

    # 2. KHỞI ĐỘNG AI THREAD
    ai_thread = AIThread(detector, infer_req, labels, out_layer)
    ai_thread.start()

    # 3. CAMERA LOOP (Chạy độc lập siêu nhanh)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(3, 1280); cap.set(4, 720)
    
    fps = 0
    prev_time = time.time()
    
    # Biến lưu kết quả hiển thị
    current_box = None
    current_text = ""

    print("---Press 'Q' to exit---")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Lật gương
            frame = cv2.flip(frame, 1)

            H, W = frame.shape[:2]
            roi_y1, roi_x1 = (H - ROI_SIZE) // 2, (W - ROI_SIZE) // 2
            roi_frame = frame[roi_y1:roi_y1+ROI_SIZE, roi_x1:roi_x1+ROI_SIZE].copy() # Copy để gửi cho AI

            # Gửi ảnh cho AI xử lý (Không chờ nó xong)
            ai_thread.update_frame(roi_frame)
            
            # Lấy kết quả mới nhất từ AI (nếu có)
            box, label, conf = ai_thread.get_result()

            # Vẽ kết quả (Luôn vẽ cái mới nhất AI tìm ra)
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x1+ROI_SIZE, roi_y1+ROI_SIZE), (200, 200, 200), 2)
            
            if box is not None:
                bx1, by1, bx2, by2 = box.astype(int)
                fx1, fy1 = bx1 + roi_x1, by1 + roi_y1
                fx2, fy2 = bx2 + roi_x1, by2 + roi_y1
                
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 3)
                cv2.putText(frame, f"{label} ({conf:.1%})", (fx1, fy1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Tính FPS
            curr_time = time.time()
            fps = 0.95*fps + 0.05*(1/(curr_time - prev_time)) if (curr_time-prev_time)>0 else 0
            prev_time = curr_time

            cv2.putText(frame, f"FPS: {fps:.1f} ", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow("GARBAGE DETECTION AND CLASSIFICATION", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # Dọn dẹp thread khi tắt
        ai_thread.stop()
        ai_thread.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()