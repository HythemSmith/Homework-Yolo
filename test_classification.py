from ultralytics import YOLO

if __name__ == '__main__':
    # Đường dẫn model tốt nhất
    model_path = 'runs/classify/cifar10_train/weights/best.pt'
    model = YOLO(model_path)
    
    # Đánh giá trên tập test
    metrics = model.val(
        data='cifar10_processed',   # cùng cấu trúc dữ liệu
        imgsz=64,
        batch=64,
        device=0,
        project='runs/classify',
        name='cifar10_test'
    )
    
    print("\n" + "="*50)
    print("KẾT QUẢ TRÊN TẬP TEST")
    print("="*50)
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")
    print(f"Confusion matrix được lưu tại: {metrics.save_dir}/confusion_matrix.png")