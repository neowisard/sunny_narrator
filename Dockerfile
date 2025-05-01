# Используем базовый образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в рабочую директорию
COPY . /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
# /app/books books dir
#-v $(pwd)/First.fb2:/app/books/ExampleBook.fb2
# Команда для запуска приложения
CMD ["python", "app.py"]