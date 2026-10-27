import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Загрузка данных
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Создаем СВЕРТОЧНУЮ нейросеть
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Первая свертка: берет 1 канал (ч-б картинка), делает 16 фильтров
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Уменьшает картинку в 2 раза (оставляет самое важное)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Вторая свертка: из 16 фильтров делает 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # После двух уменьшений (pool) картинка 28x28 стала 7x7.
        # 32 фильтра * 7 * 7 пикселей = 1568. Отдаем это в обычную сеть.
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # Свертка + активация + сжатие
        x = self.pool(self.relu(self.conv2(x))) # Вторая свертка + сжатие
        x = x.view(-1, 32 * 7 * 7)              # Разворачиваем в линию
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Инициализация
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Обучение
print("Начинаем обучение CNN...")
epochs = 3 # CNN учится быстрее и лучше, 3 эпох для примера достаточно
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Эпоха {epoch+1} завершена. Ошибка: {loss.item():.4f}")

# 5. Проверка
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"\nТочность CNN на тестовых данных: {accuracy:.2f}%")

# 6. Сохраняем новую модель
torch.save(model.state_dict(), "cnn_digit_model.pth")
print("Модель сохранена в cnn_digit_model.pth")