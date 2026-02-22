from torchvision import datasets, transforms
from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from model import MLP,CNN

transform = transforms.ToTensor()

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

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

def training_loop():
    model = MLP()
    model = CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            # print(images.shape)  # (batch_size, 1, 28, 28)
            # print(labels.shape)  # (batch_size,)
            y_logits = model(images)
            # print(y_logits.shape)
            # y_pred = torch.argmax(y_logits, dim=1)
            # print(labels.shape)
            loss = loss_fn(y_logits,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss: {loss}")

    torch.save(model.state_dict(), "mnist_model_cnn.pth")
    print("Model saved!")
    return model

def test_loop(model):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            total_loss += loss.item()
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")


if __name__=="__main__":
    model = training_loop()
    test_loop(model)

