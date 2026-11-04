
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Device:", DEVICE)


# ==========================================
# 2. LOAD MNIST DATASET
# ==========================================

transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get one batch containing 4 images
images, labels = next(iter(loader))

images = images.to(DEVICE)
labels = labels.to(DEVICE)

print("\nINPUT BATCH:", images.shape)
print("LABELS:", labels.tolist())


# ==========================================
# 3. BUILD CNN MODEL
# ==========================================

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolution layer 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # Convolution layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        results = {}

        # Input
        results["Input"] = x

        # Conv1
        x = self.conv1(x)
        results["Conv1"] = x

        x = self.relu1(x)
        results["ReLU1"] = x

        x = self.pool1(x)
        results["MaxPool1"] = x

        # Conv2
        x = self.conv2(x)
        results["Conv2"] = x

        x = self.relu2(x)
        results["ReLU2"] = x

        x = self.pool2(x)
        results["MaxPool2"] = x

        # Flatten
        x = self.flatten(x)
        results["Flatten"] = x

        # Fully Connected 1
        x = self.fc1(x)
        x = self.relu3(x)
        results["FC1"] = x

        # Output
        x = self.fc2(x)
        results["Output"] = x

        return x, results


# ==========================================
# 4. FORWARD PASS
# ==========================================

model = SimpleCNN().to(DEVICE)
model.eval()

with torch.no_grad():

    logits, results = model(images)

    probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(
        probabilities,
        dim=1
    )


# ==========================================
# 5. PRINT TENSOR SHAPES
# ==========================================

print("\n========== TENSOR SHAPES ==========")

for name, tensor in results.items():
    print(f"{name:12s}: {list(tensor.shape)}")


# ==========================================
# 6. PRINT PREDICTIONS
# ==========================================

print("\n========== PREDICTIONS ==========")

for i in range(BATCH_SIZE):

    print(
        f"Sample {i+1}: "
        f"Actual = {labels[i].item()}, "
        f"Predicted = {predictions[i].item()}"
    )


# ==========================================
# 7. VISUALIZE INPUT BATCH
# ==========================================

fig, axes = plt.subplots(1, 4, figsize=(12, 3))

fig.suptitle("INPUT BATCH - 4 MNIST SAMPLES")

for i in range(4):

    axes[i].imshow(
        images[i, 0].cpu(),
        cmap="gray"
    )

    axes[i].set_title(
        f"Sample {i+1}\nLabel: {labels[i].item()}"
    )

    axes[i].axis("off")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "01_input_batch.png", dpi=150)


# ==========================================
# 8. VISUALIZE FEATURE MAPS
# ==========================================

layers = [
    ("Conv1", results["Conv1"]),
    ("MaxPool1", results["MaxPool1"]),
    ("Conv2", results["Conv2"]),
    ("MaxPool2", results["MaxPool2"])
]

for layer_name, feature_maps in layers:

    fig, axes = plt.subplots(
        4, 4,
        figsize=(10, 9)
    )

    fig.suptitle(
        f"{layer_name} - Feature Maps",
        fontsize=16
    )

    for sample in range(4):

        for channel in range(4):

            axes[sample, channel].imshow(
                feature_maps[sample, channel].cpu(),
                cmap="viridis"
            )

            axes[sample, channel].set_title(
                f"Sample {sample+1} - Channel {channel+1}"
            )

            axes[sample, channel].axis("off")

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"02_{layer_name}.png",
        dpi=150
    )


# ==========================================
# 9. VISUALIZE FLATTEN AND FC
# ==========================================

flatten_data = results["Flatten"].cpu()
fc_data = results["FC1"].cpu()

fig, axes = plt.subplots(2, 1, figsize=(12, 5))

fig.suptitle("FLATTEN AND FULLY CONNECTED")

axes[0].imshow(
    flatten_data,
    aspect="auto",
    cmap="viridis"
)

axes[0].set_title("Flatten: [4, 1568]")
axes[0].set_ylabel("Samples")

axes[1].imshow(
    fc_data,
    aspect="auto",
    cmap="viridis"
)

axes[1].set_title("FC1: [4, 128]")
axes[1].set_ylabel("Samples")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "03_flatten_fc.png", dpi=150)


# ==========================================
# 10. VISUALIZE PREDICTIONS
# ==========================================

fig, axes = plt.subplots(
    1, 4,
    figsize=(14, 4)
)

fig.suptitle("CNN FORWARD PASS - PREDICTION RESULTS")

for i in range(4):

    axes[i].bar(
        range(10),
        probabilities[i].cpu().numpy()
    )

    axes[i].set_title(
        f"Sample {i+1}\n"
        f"True: {labels[i].item()} | "
        f"Pred: {predictions[i].item()}"
    )

    axes[i].set_xlabel("Digit")
    axes[i].set_ylabel("Probability")
    axes[i].set_xticks(range(10))

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "04_predictions.png", dpi=150)


# ==========================================
# 11. SUMMARY
# ==========================================

print("\n========== FORWARD PASS COMPLETED ==========")

print("Input shape:", list(images.shape))
print("Output shape:", list(logits.shape))

print("\nGenerated visualization files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file)

print("\nCNN Forward Pass completed successfully!")

# Display all visualization windows
plt.show()
