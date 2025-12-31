import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st

MODEL_PATH = os.path.join("models", "attention_cnn.pth")
IMAGE_SIZE = 224

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def main():
    st.title("Student Attention Detection System")
    st.write("Upload an image of a student and classify attention state.")

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please run train_cnn.py first.")
        return

    model, classes = load_model()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)



        if st.button("Predict Attention State"):
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze().tolist()

            max_idx = int(torch.argmax(outputs, dim=1).item())
            predicted_class = classes[max_idx]
            confidence = probs[max_idx] * 100.0

            st.subheader("Prediction")
            st.write(f"Class: **{predicted_class}**")
            st.write(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()
