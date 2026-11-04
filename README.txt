CNN FORWARD PASS VISUALIZATION

Dataset: MNIST
Batch Size: 4

CNN Architecture:
Input -> Conv1 -> ReLU -> MaxPool
-> Conv2 -> ReLU -> MaxPool
-> Flatten -> Fully Connected -> Output

Tensor Shapes:
Input: [4, 1, 28, 28]
Conv1: [4, 16, 28, 28]
MaxPool1: [4, 16, 14, 14]
Conv2: [4, 32, 14, 14]
MaxPool2: [4, 32, 7, 7]
Flatten: [4, 1568]
FC1: [4, 128]
Output: [4, 10]

How to run:
pip install -r requirements.txt
python cnn_forward.py

Output:
7 visualization images in the visualizations folder.

Note:
The model has not been trained.
Predictions are for forward-pass demonstration only.