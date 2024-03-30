# UfaHack2024
<img src="https://images.squarespace-cdn.com/content/v1/6201920e5d62b32f53d158bb/1646055024414-55O3Y8GPH2DM30WXAWVM/Screen%2BShot%2B2021-11-18%2Bat%2B1.18.40%2BPM.png" alt="Face" width="10%" height="10%">
Hackaton UfaHack2024 in UFA

____

## Реализация модели
![f](https://github.com/SKYLIGHTSUFA/UfaHack2024/blob/main/c4715817-515e-4815-aa0d-bfcc75d45388.jfif)


## Программный продукт создан с помощью фреймворка для Python Socket, CatBoost, CustomTkinter, а также приложение написанное на Kotin для клиента на Android

+ Реализация с CUDA технологиями    
+ Андройд приложение    
+ Классифкатор на основе градиентного бустинга    
+ Очищенный и подготовленный датасет    
+ Возможность работать удаленно    
+ Возможность обработки в реальном времени    

## Gold features  
+ Алгоритм определения "замыленных" фотографий opencv методами
+ Очистка датасета от фотографий, где присутствует более 2 человек
+ Пакетная обработка видео, ускоряющее обнаружениие лица на одном кадре за 5ms
+ Выделение face embeddings моделью facenet, которые используются для классификации человека алгоритмом градиентного бустинга "catboost"
+ Обучение нескольких моделей для разных категорий  
## Установка зависимостей    
```
pip install catboost, customtkinter, deepface, socket, opencv-python, opencv-contrib-python, pandas, imutils, mtcnn
```
## Quick start    

``` git clone https://github.com/SKYLIGHTSUFA/UfaHack2024.git    
cd UfaHack2024 
cd notebooks/apps
python main.py
```
  

### Код реализации сервера 
```python
import socket

s = socket.socket()
host = "192.168.120.244"
port = 12345
s.bind((host, port))
s.listen(5)

while True:
    con, addr = s.accept()
    with open('FDJ.jpg', 'wb') as f:
        while True:
            print(1)
            data = con.recv(4096)
            if not data:
                break
            f.write(data)
    con.close()

```

### Код реализации клиента
```python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.connect(('192.168.120.244', 12345)) # Подключаемся к серверу.
s.sendall('Hello, Habr!'.encode('utf-8')) # Отправляем фразу.
data = s.recv(1024) # Получаем данные из сокета.
print(data.decode())
while True:
    data = s.recv(4096)
    if not data:
        break
    print("Received response: " + data.decode())

s.close()
```
Для обучения модели исползовали [CatBoost](https://catboost.ai/)

