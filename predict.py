import sys
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from model import MNISTCNN


def load_image(image_path):
    image = Image.open(image_path).convert("L")  # 灰度图

    # 反色处理：如果你的图片是白底黑字，MNIST通常是黑底白字，需要反转
    image = ImageOps.invert(image)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = transform(image)
    image = image.unsqueeze(0)  # [1, 1, 28, 28]
    return image


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load("checkpoints/mnist_cnn.pth", map_location=device))
    model.eval()

    image = load_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    print(f"Predicted digit: {pred}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict(image_path)
