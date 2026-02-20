import streamlit as st
import torch
from models.NN.NN import NeuralNetwork
from configs.utils import get_config
import cv2
import numpy as np
from PIL import Image

# Load config
config = get_config('configs/exp_1.yaml')

# Load model
model = NeuralNetwork(config.model)
ckpt = torch.load('checkpoints/model_exp_1.pth', map_location='cpu')
model.load_state_dict(ckpt)
model.eval()

st.title('Real-time Cat vs Dog Classifier')

FRAME_WINDOW = st.image([])
run = st.checkbox('Bật camera')

cap = None
if run:
    cap = cv2.VideoCapture(0)
    st.write('Camera đang bật...')
else:
    st.write('Camera đã tắt.')

while run:
    ret, frame = cap.read()
    if not ret:
        st.write('Không lấy được frame từ camera!')
        break
    # Dự đoán trên ảnh resize, hiển thị label trên frame gốc nét nhất
    img_for_model = cv2.resize(frame, (config.model.width, config.model.height))
    img_for_model = cv2.cvtColor(img_for_model, cv2.COLOR_BGR2RGB)
    import time
    img_tensor = torch.tensor(img_for_model, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0
    start_time = time.time()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred = pred.item()
        conf = conf.item()
    elapsed = (time.time() - start_time) * 1000  # ms
    label = 'Cat' if pred == 0 else 'Dog'
    # Hiển thị kết quả trên frame gốc nét
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    text = f"{label} ({conf*100:.1f}%) | {elapsed:.1f} ms"
    frame_disp = cv2.putText(frame_rgb.copy(), text, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)
    FRAME_WINDOW.image(frame_disp)

if cap:
    cap.release()