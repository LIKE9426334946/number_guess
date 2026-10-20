import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()  # 总参数量：421642
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 参数量：288+32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 参数量：18432+64
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 参数量：401408+128
        self.fc2 = nn.Linear(128, 10)  # 参数量：1280+10

    def forward(self, x):  # x.shape=torch.Size([64, 1, 28, 28])

        x = self.pool(F.relu(self.conv1(x)))  # x.shape=torch.Size([64, 32, 14, 14])
        x = self.pool(F.relu(self.conv2(x)))  # x.shape=torch.Size([64, 64, 7, 7])
        x = torch.flatten(x, 1)  # 64*7*7,x.shape=torch.Size([64, 3136])
        x = F.relu(self.fc1(x))  # x.shape=torch.Size([64, 128])
        x = self.fc2(x)  # x.shape=torch.Size([64, 10])

        return x


if __name__ == "__main__":
    model = MNISTCNN()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量：{trainable_params}")
