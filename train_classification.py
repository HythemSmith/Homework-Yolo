from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s-cls.pt')  # dùng model small cho classification
    model.train(
        data='cifar10_processed',   # đường dẫn đến thư mục chứa train/ và test/
        epochs=10,
        imgsz=64,                   # resize ảnh lên 64x64
        batch=64,
        device=0,                   # GPU 0 (nếu không có GPU thì đổi 'cpu')
        project='runs/classify',
        name='cifar10_train'
    )