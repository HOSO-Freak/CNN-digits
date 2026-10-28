import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. Описываем архитектуру (точно такую же, как при обучении)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Загружаем модель
model = CNN()
model.load_state_dict(torch.load("cnn_digit_model.pth"))
model.eval()

# 3. Готовим картинку
# MNIST обучался на черных цифрах с белым фоном.
# Мы переводим картинку в ч/б, меняем размер на 28x28 и превращаем в тензор.
from torchvision import transforms
from PIL import ImageOps # Нам понадобится эта штука для инверсии

# Загружаем картинку
image = Image.open("3.png").convert('L') # Открываем и делаем серой
image = ImageOps.invert(image)           # ИНВЕРСИЯ (делаем белым на черном)

transform = transforms.Compose([
    transforms.Resize((28, 28)), # Сжимаем
    transforms.ToTensor(),       # В тензор
    transforms.Normalize((0.5,), (0.5,)) # Та же нормализация, что при обучении
])

input_tensor = transform(image).unsqueeze(0)

# 4. Показываем картинку, которую видит сеть
plt.imshow(input_tensor.squeeze().numpy(), cmap='gray')
plt.title("Твоя цифра (как её видит сеть)")
plt.show()

# 5. Предсказание
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

print(f"Нейросеть думает, что ты нарисовал: {predicted.item()}")