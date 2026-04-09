import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import gdown
import os

# 1. Redefine the exact same architecture used in Colab
class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.rnn = nn.LSTM(512 * 8, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv_out = self.cnn(x)
        b, c, h, w = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2).contiguous()
        conv_out = conv_out.view(b, w, c * h)
        rnn_out, _ = self.rnn(conv_out)
        output = self.fc(rnn_out)
        return output.permute(1, 0, 2)

# 2. Loading Function with Google Drive Bypass & Caching
@st.cache_resource
def load_model():
    model_path = 'washington_crnn.pth'
    vocab_path = 'vocab.pth'
    
    # NEW: Download model from Google Drive if it's missing locally
    if not os.path.exists(model_path):
        with st.spinner("Downloading model weights... (This may take a minute on first run)"):
      
            file_id = '13GMypOiKeVgasvnCSPd6ta32GRFu91IL' 
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error(f"Failed to download model from Google Drive: {e}")
                return None, None

    try:
        vocab_data = torch.load(vocab_path, map_location='cpu')
        idx_to_char = vocab_data['idx_to_char']
        num_classes = len(idx_to_char) + 1
        
        model = CRNN(num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, idx_to_char
    except Exception as e:
        st.error(f"Failed to load model or vocab files. Error: {e}")
        return None, None

# 3. Image Preprocessing
def preprocess_image(image):
    # Convert PIL to OpenCV grayscale
    img = np.array(image.convert('L'))
    
    # Same padding logic as training
    target_h, target_w = 64, 800
    h, w = img.shape
    scale = target_h / h
    new_w = min(int(w * scale), target_w)
    img = cv2.resize(img, (new_w, target_h))
    
    padded_img = np.ones((target_h, target_w), dtype=np.uint8) * 255
    padded_img[:, :new_w] = img
    
    padded_img = (padded_img / 127.5) - 1.0
    tensor = torch.FloatTensor(padded_img).unsqueeze(0).unsqueeze(0)
    return tensor

# 4. CTC Greedy Decoder
def decode_predictions(preds, idx_to_char):
    # preds shape: [Seq_len, Batch, Classes] -> [Seq_len, Classes]
    preds = preds[:, 0, :]
    _, max_indices = torch.max(preds, 1)
    
    # Greedy decoding: remove duplicates and 0 (blank token)
    decoded = []
    prev_idx = -1
    for idx in max_indices.tolist():
        if idx != 0 and idx != prev_idx:
            decoded.append(idx_to_char[idx])
        prev_idx = idx
        
    return "".join(decoded)

# 5. Streamlit User Interface
st.set_page_config(page_title="Historical OCR", page_icon="📜", layout="centered")

st.title("📜 Historical Handwriting Recognition")
st.markdown("Upload a binarized line-level image from the Washington Database to transcribe it.")

# Load resources
model, idx_to_char = load_model()

if model is not None:
    uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Document Line", use_container_width=True)
        
        if st.button("Transcribe Document", type="primary"):
            with st.spinner('Analyzing handwriting sequences...'):
                tensor_img = preprocess_image(image)
                
                with torch.no_grad():
                    preds = model(tensor_img)
                    
                transcription = decode_predictions(preds, idx_to_char)
                
            st.success("Transcription Complete!")
            st.code(transcription, language='text')

st.sidebar.markdown("""
### About This App
Developed for the **Group 4 DeepTech Capstone**. 
This utilizes a PyTorch CRNN + CTC Loss architecture trained specifically on the 18th-century George Washington historical manuscript dataset.
""")
