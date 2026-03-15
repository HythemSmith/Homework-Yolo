import os
import tarfile
import pickle
import numpy as np
from PIL import Image
import shutil

# Đường dẫn file tar.gz (nếu để cùng thư mục thì chỉ cần tên file)
tar_path = 'cifar-10-python.tar.gz'
output_dir = 'cifar10_processed'

# Tên các lớp theo thứ tự từ 0 đến 9
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def extract_cifar10(tar_path, output_dir):
    # Giải nén tar.gz
    extract_path = 'cifar-10-batches-py'
    if not os.path.exists(extract_path):
        print("Đang giải nén...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall()
    else:
        print("Đã có thư mục giải nén, bỏ qua.")

    # Tạo thư mục output (xóa nếu đã tồn tại)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'train'))
    os.makedirs(os.path.join(output_dir, 'test'))

    # Hàm đọc file pickle của CIFAR
    def unpickle(file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    # Xử lý 5 batch train (data_batch_1 đến data_batch_5)
    print("Đang xử lý tập train...")
    train_dir = os.path.join(output_dir, 'train')
    for i in range(1, 6):
        batch_file = os.path.join(extract_path, f'data_batch_{i}')
        batch = unpickle(batch_file)
        data = batch[b'data']       # shape (10000, 3072)
        labels = batch[b'labels']    # list 10000 số
        for j, (img_data, label) in enumerate(zip(data, labels)):
            # Chuyển đổi dữ liệu về ảnh 32x32x3
            img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
            img_pil = Image.fromarray(img)
            class_dir = os.path.join(train_dir, class_names[label])
            os.makedirs(class_dir, exist_ok=True)
            # Đặt tên file ảnh theo số thứ tự tổng
            img_pil.save(os.path.join(class_dir, f'{(i-1)*10000 + j}.png'))
        print(f"  Đã xử lý batch {i}")

    # Xử lý batch test
    print("Đang xử lý tập test...")
    test_dir = os.path.join(output_dir, 'test')
    test_batch_file = os.path.join(extract_path, 'test_batch')
    test_batch = unpickle(test_batch_file)
    data = test_batch[b'data']
    labels = test_batch[b'labels']
    for j, (img_data, label) in enumerate(zip(data, labels)):
        img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img_pil = Image.fromarray(img)
        class_dir = os.path.join(test_dir, class_names[label])
        os.makedirs(class_dir, exist_ok=True)
        img_pil.save(os.path.join(class_dir, f'{j}.png'))
    print("Hoàn tất!")

if __name__ == '__main__':
    extract_cifar10(tar_path, output_dir)
    print(f"Dữ liệu đã được chuẩn bị tại thư mục '{output_dir}'")