import os

from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv

if __name__ == '__main__':
    
    # 1. TẢI DATASET TỪ ROBOFLOW
    load_dotenv()
    rf = Roboflow(api_key=os.getenv("API_KEY"))
    project = rf.workspace("adip-xskmh").project("homework-c85ln")
    version = project.version(1)
    dataset = version.download("yolov8")

    # 2. KHỞI TẠO MÔ HÌNH YOLOv8 SEGMENTATION
    model = YOLO('yolov8s.pt')

    # 3. TIẾN HÀNH HUẤN LUYỆN (TRAINING)
    results = model.train(
        data=f"{dataset.location}/data.yaml", 
        epochs=100,
        imgsz=640,
        batch=16,
        device=0, 
        project="runs/object_detection",
        name="train_results",
        workers=2 # Thêm thông số này để load data nhẹ nhàng hơn trên máy cá nhân
    )

    print("Đã huấn luyện xong! Mở thư mục runs/object detection/train_results để xem biểu đồ.")
                