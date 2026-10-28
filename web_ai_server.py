import torch
import torch.nn as nn
from http.server import BaseHTTPRequestHandler, HTTPServer
from torchvision import transforms
from PIL import ImageOps, Image
import io
import urllib.parse
import base64

# 1. Описываем нашу CNN (ту же, что и при обучении)
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

# Загружаем модель в память сервера
model = CNN()
model.load_state_dict(torch.load("cnn_digit_model.pth"))
model.eval()

class AIServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        
        # Простая HTML-страница с формой загрузки
        html = """
        <html>
            <head><title>AI Распознавание цифр</title></head>
            <body style="text-align: center; font-family: sans-serif; margin-top: 50px;">
                <h1>Загрузи картинку с цифрой</h1>
                <form action="/" method="POST" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/png, image/jpeg">
                    <input type="submit" value="Отправить нейросети">
                </form>
            </body>
        </html>
        """
        self.wfile.write(html.encode('utf-8'))

    def do_POST(self):
        # 1. Получаем длину данных
        content_length = int(self.headers['Content-Length'])
        # 2. Читаем данные из тела запроса
        post_data = self.rfile.read(content_length)
        
        # 3. Парсим multipart/form-data (это то, как передаются файлы в вебе)
        boundary = self.headers['Content-Type'].split("=")[1]
        parts = post_data.split(b'--' + boundary.encode())
        
        image_data = None
        for part in parts:
            if b'filename=' in part:
                # Находим, где начинаются сами байты картинки (после \r\n\r\n)
                file_bytes = part.split(b'\r\n\r\n')[1]
                # Отрезаем последние два байта переноса строки
                image_data = file_bytes[:-2]
                break

        if not image_data:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"File not found")
            return

        # 4. Обрабатываем картинку
        image = Image.open(io.BytesIO(image_data)).convert('L')
        image = ImageOps.invert(image) # Инверсия (белое на черном)
        
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        input_tensor = transform(image).unsqueeze(0)

        # 5. Нейросеть делает предсказание
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            digit = predicted.item()

        # 6. Возвращаем ответ в браузер
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        
        # Кодируем картинку в base64, чтобы показать её прямо в HTML
        img_base64 = base64.b64encode(image_data).decode('utf-8')
        
        response = f"""
        <html>
            <body style="text-align: center; font-family: sans-serif; margin-top: 50px;">
            <h1>Нейросеть думает, что это цифра: <span style="color: red; font-size: 2em;">{digit}</span></h1>
                <img src="data:image/png;base64,{img_base64}" style="height: 200px; margin: 20px;">
                <br><br>
                <a href="/">Загрузить другую</a>
            </body>
        </html>
        """
        self.wfile.write(response.encode('utf-8'))

# Запуск сервера
port = 8080
server = HTTPServer(('localhost', port), AIServer)
print(f"Сервер с ИИ запущен на http://localhost:{port}")
print("Нажми Ctrl+C, чтобы остановить.")
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nСервер остановлен.")