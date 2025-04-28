from flask import Flask, request, jsonify
import numpy as np
import pymysql
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

app = Flask(__name__)

# Cấu hình MySQL
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
}

def load_products_from_mysql():
    try:
        connection = pymysql.connect(**MYSQL_CONFIG)
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("SELECT id, title, category, origin, description, price, ocop_star FROM products")
            products = cursor.fetchall()
        connection.close()
        print("✅ Loaded products from MySQL")
        return products
    except Exception as e:
        print(f"⚠️ Error connecting MySQL: {e}")
        return None
    
def parse_price(price_str):
    # Xóa dấu chấm và chữ 'đ'
    clean_price = price_str.replace('.', '').replace('đ', '').strip()
    try:
        return int(clean_price)
    except ValueError:
        return 0
    
def load_products_from_csv(file_path):
    products = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, row in enumerate(reader):
                products.append({
                    "id": idx + 1,  # tự sinh id
                    "title": row["product_name"],
                    "category": row["category"],
                    "origin": row["location"],
                    "description": row["description"],
                    "price": parse_price(row["price"]),
                    "ocop_star": int(row["star"])
                })
        print("✅ Loaded products from CSV")
        return products
    except Exception as e:
        print(f"⚠️ Error loading CSV: {e}")
        return []

# ========= Load products ==========

products = load_products_from_mysql()

# Nếu MySQL lỗi thì fallback qua CSV
if not products:
    if os.path.exists('product.csv'):
        products = load_products_from_csv('product.csv')
    else:
        print("❌ Không tìm thấy data MySQL hoặc CSV!")
        products = []

# ========= Chuẩn bị TF-IDF ==========

product_texts = []
product_id_to_index = {}

for idx, p in enumerate(products):
    full_text = f"{p['title']} {p['category']} {p['origin']} {p['description']}"
    product_texts.append(full_text)
    product_id_to_index[p["id"]] = idx

vectorizer = TfidfVectorizer()
product_vectors = vectorizer.fit_transform(product_texts)

# ========= API Recommend ==========

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    
    search_history = data.get("search_history", [])
    viewed_product_ids = data.get("viewed_product_ids", [])
    top_k = data.get("top_k", 5)

    if not search_history and not viewed_product_ids:
        return jsonify({"error": "Cần ít nhất search_history hoặc viewed_product_ids"}), 400

    # Vector từ search history
    if search_history:
        search_vectors = vectorizer.transform(search_history)
        mean_search_vector = np.mean(search_vectors.toarray(), axis=0)
    else:
        mean_search_vector = np.zeros(product_vectors.shape[1])

    # Vector từ sản phẩm đã xem
    view_vectors = []
    for pid in viewed_product_ids:
        idx = product_id_to_index.get(pid)
        if idx is not None:
            view_vectors.append(product_vectors[idx])

    if view_vectors:
        view_vectors = np.vstack([v.toarray() for v in view_vectors])
        mean_view_vector = np.mean(view_vectors, axis=0)
    else:
        mean_view_vector = np.zeros(product_vectors.shape[1])

    # Trọng số
    search_weight = 0.7
    view_weight = 0.3

    user_vector = (
        search_weight * mean_search_vector +
        view_weight * mean_view_vector
    ).reshape(1, -1)

    similarities = cosine_similarity(user_vector, product_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    recommended_products = []
    for idx in top_indices:
        p = products[idx]
        recommended_products.append({
            "id": p["id"],
            "title": p["title"],
            "category": p["category"],
            "origin": p["origin"],
            "description": p["description"],
            "price": f"{p['price']:,}đ",  # format lại giá đẹp 180,000đ
            "ocop_star": p["ocop_star"],
            "score": round(float(similarities[idx]), 4)
        })

    return jsonify({
        "recommended_products": recommended_products
    })

# ========= RUN APP ==========

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

#{
#  "search_history": [
#    "trà xanh thái nguyên",
#    "mật ong nguyên chất tự nhiên"
#  ],
#  "viewed_product_ids": [1, 2, 5],
#  "top_k": 5
#}


 