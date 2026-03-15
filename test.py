import os
import glob
from ultralytics import YOLO

model = YOLO('runs/train/train_results/weights/best.pt')

test_images_path = 'object_detection_yolo/test/images' 

image_files = glob.glob(os.path.join(test_images_path, '*.jpg'))

print(f"Tìm thấy {len(image_files)} ảnh trong tập test. Đang thực hiện nhận diện...")

for img_path in image_files[:5]: 
    results = model.predict(source=img_path, save=True, conf=0.5)
    print(f"Đã lưu kết quả của: {os.path.basename(img_path)}")

print("Xong! Kiểm tra thư mục 'runs/object_detection/predict' để lấy ảnh/video dán vào báo cáo.")
