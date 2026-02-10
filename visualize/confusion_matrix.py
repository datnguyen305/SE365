import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None, title='Confusion Matrix'):
    """
    Vẽ và lưu Confusion Matrix.
    Args:
        y_true: Mảng nhãn thật
        y_pred: Mảng nhãn dự đoán
        labels: Danh sách tên các lớp (ví dụ: ['Cat', 'Dog'])
        save_path: Đường dẫn để lưu file ảnh (ví dụ: 'checkpoints/cm.png')
        title: Tiêu đề của biểu đồ
    """
    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Tính toán tỷ lệ phần trăm (normalized)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    
    # Tạo nội dung hiển thị trong từng ô (Số lượng \n Tỷ lệ %)
    annot = np.array([["{0:d}\n({1:.1f}%)".format(count, perc) 
                       for count, perc in zip(row_count, row_perc)] 
                      for row_count, row_perc in zip(cm, cm_perc)])

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=15)
    
    if save_path:
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f" -> Biểu đồ đã được lưu tại: {save_path}")
    
    plt.show()


