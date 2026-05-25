from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "best_eb0_se_sam_gap_sn.pth"
# Change order here if your training label mapping was different.
CLASS_NAMES = ["Benign", "Malignant"]


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class EfficientNetSE(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        self.features = backbone.features
        in_channels = backbone.classifier[1].in_features
        self.se = SEBlock(in_channels)
        self.sam = SpatialAttentionModule(kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Uploaded model has keys: classifier.1.weight and classifier.1.bias
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.sam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_model():
    model = EfficientNetSE(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(path):
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx].item())

    return label, round(confidence, 3)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        files = request.files.getlist("files")

        for f in files:
            if not f or f.filename == "":
                continue
            filename = secure_filename(f.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            f.save(filepath)
            predicted, conf = predict_image(filepath)
            # No actual label is available from uploaded image, so Actual is shown same as prediction.
            # If you later upload images from folders/classes, this can be changed.
            actual = predicted
            results.append((filename, actual, predicted, conf))

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
