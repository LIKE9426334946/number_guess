import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import MNISTCNN


def setup_ddp():
    # torchrun 会自动注入这些环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 单机多卡 GPU 训练通常使用 nccl
    dist.init_process_group(backend="nccl")

    # 让当前进程只使用对应的 GPU
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def train():
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"DDP started. world_size={world_size}, local_rank={local_rank}")

    # 这里的 batch_size 是“每张卡”的 batch size
    batch_size = 256
    learning_rate = 1e-3
    epochs = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # 关键：DDP 下必须给训练集配 DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # 测试集也可以用 DistributedSampler，这样每个进程只评估一部分
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    model = MNISTCNN().to(device)

    # 关键：先 .to(device)，再包 DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # 关键：每个 epoch 设置一次，保证多进程 shuffle 正常
        train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        # 当前进程上的局部指标
        local_train_loss = torch.tensor(running_loss, device=device)
        local_train_correct = torch.tensor(train_correct, device=device)
        local_train_total = torch.tensor(train_total, device=device)

        # all_reduce 聚合所有进程的统计量
        dist.all_reduce(local_train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_train_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_train_total, op=dist.ReduceOp.SUM)

        avg_loss = local_train_loss.item() / len(train_loader) / world_size
        train_acc = 100.0 * local_train_correct.item() / local_train_total.item()

        # 验证
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                preds = outputs.argmax(dim=1)
                test_total += labels.size(0)
                test_correct += (preds == labels).sum().item()

        local_test_correct = torch.tensor(test_correct, device=device)
        local_test_total = torch.tensor(test_total, device=device)

        dist.all_reduce(local_test_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_test_total, op=dist.ReduceOp.SUM)

        test_acc = 100.0 * local_test_correct.item() / local_test_total.item()

        # 只在主进程打印，避免重复
        if rank == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {avg_loss:.4f} "
                f"Train Acc: {train_acc:.2f}% "
                f"Test Acc: {test_acc:.2f}%"
            )

    # 只在主进程保存
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        save_path = "checkpoints/mnist_cnn.pth"
        torch.save(model.module.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    cleanup_ddp()


if __name__ == "__main__":
    train()
