# 1. Dùng Python image chính thức
FROM python:3.11-slim

# 2. Set thư mục làm việc trong container
WORKDIR /app

# 3. Copy requirements.txt và install các thư viện
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy toàn bộ mã nguồn app vào container
COPY . .

# 5. Copy luôn file .env nếu bạn cần (nếu không tự handle trong docker-compose)

# 6. Expose port Flask (mặc định 5000)
EXPOSE 5000

# 7. Run Flask app
CMD ["python", "app.py"]
