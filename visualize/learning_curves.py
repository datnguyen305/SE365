import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curves(csv_path, save_path=None):
    """
    Đọc file log CSV và vẽ biểu đồ Loss/Accuracy.
    Args:
        csv_path: Đường dẫn file training_log_xxx.csv
        save_path: Đường dẫn lưu file ảnh biểu đồ
    """
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file log tại {csv_path}")
        return

    df = pd.read_csv(csv_path)
    epochs = df['epoch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Biểu đồ Loss
    ax1.plot(epochs, df['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, df['dev_loss'], label='Dev Loss', marker='s')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Biểu đồ Accuracy
    ax2.plot(epochs, df['dev_accuracy'], label='Dev Accuracy', color='green', marker='o')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f" -> Learning curves saved at: {save_path}")
    
    plt.show()