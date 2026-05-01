import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

# ---------------------------
# 1. PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="CIFAR-10 Vision AI", layout="centered")

# ---------------------------
# 2. LOAD CSS
# ---------------------------
def local_css(file_name):
    # Create the path relative to THIS file
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, file_name)
    
    if os.path.exists(full_path):
        with open(full_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# ---------------------------
# 3. MODEL ARCHITECTURE
# ---------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ---------------------------
# 4. LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model = SimpleNN()
    
    # Create the path relative to THIS file (app.py)
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "cifar10_cnn.pth")
    
    # Use the dynamic path
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --- ADD THIS LINE ---
model = load_model()

# ---------------------------
# 5. HEADER UI
# ---------------------------
st.markdown("""
<div class="main-container">
<h1>Vision AI Classifier</h1>
<p class="subtitle">Deep Learning solution for CIFAR-10 image recognition.</p>
<div class="stats-grid">
<div class="stat-card"><span>Accuracy</span><strong>80.11%</strong></div>
<div class="stat-card"><span>Framework</span><strong>PyTorch</strong></div>
<div class="stat-card"><span>Epochs</span><strong>20</strong></div>
</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 6. IMAGE UPLOAD
# ---------------------------
st.markdown("### Upload your image")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    img_tensor = transform(image).unsqueeze(0)

    # ---------------------------
    # 7. PREDICTION
    # ---------------------------
    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, 1)

    predicted_class = classes[prediction.item()]
    confidence_score = confidence.item() * 100

    st.markdown(f"""
    <div class="result-box">
        <h2>{predicted_class.upper()}</h2>
        <p>Confidence: {confidence_score:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# 8. FOOTER
# ---------------------------
st.markdown("""
<div class="footer-note">
    Work hard, never take a rest, and keep your eyes on your goal.
</div>
""", unsafe_allow_html=True)