# UfaHack2024
<img src="https://images.squarespace-cdn.com/content/v1/6201920e5d62b32f53d158bb/1646055024414-55O3Y8GPH2DM30WXAWVM/Screen%2BShot%2B2021-11-18%2Bat%2B1.18.40%2BPM.png" alt="Face" width="10%" height="10%">
Hackaton UfaHack2024 in UFA

____

## Данный программный продукт создан с помощью фреймворка для Python Socket, CutBoost, CustomTkinter, а также приложение написанное на Kotin для клиента на Android

### Код реализации сервера 
``` 
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
            data = con.recv(12000000)
            if not data:
                break
            f.write(data)
    con.close()

```

### Код реализации клиента
```
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
Для обучения модели исползовали [CutBoost](https://catboost.ai/)
