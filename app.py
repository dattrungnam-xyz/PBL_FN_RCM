from flask import Flask, request, jsonify
import numpy as np
import pymysql
import pandas as pd
import os
import csv
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
app = Flask(__name__)

# ========== Cấu hình MySQL ==========
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
        print(f"⚠️ MySQL error: {e}")
        return None

def parse_price(price_str):
    # Nếu giá là chuỗi trống, trả về 0
    if not price_str:
        return 0
    # Xóa dấu chấm và chữ 'đ'
    clean_price = price_str.replace('.', '').replace('đ', '').strip()
    try:
        return int(clean_price)
    except ValueError:
        # Nếu không chuyển đổi được, trả về 0
        return 0


def load_products_from_csv(file_path):
    products = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                products.append({
                    "id": idx + 1,
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
        print(f"⚠️ CSV error: {e}")
        return []

# ========== Load Products ==========
products = load_products_from_mysql()
if not products and os.path.exists('product.csv'):
    products = load_products_from_csv('product.csv')

if not products:
    raise Exception("❌ No product data found")

# ========== Prepare Data ==========
product_texts = []
product_id_to_index = {}

for idx, p in enumerate(products):
    full_text = f"{p['title']} {p['category']} {p['origin']} {p['description']}"
    product_texts.append(full_text)
    product_id_to_index[p['id']] = idx

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(product_texts)

# ========== Recommend API ==========
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    search_history = data.get("search_history", [])
    viewed_product_ids = data.get("viewed_product_ids", [])
    current_search = data.get("current_search", "")
    origin_filter = data.get("origin")
    min_price = data.get("min_price", 0)
    max_price = data.get("max_price", float("inf"))
    top_k = data.get("top_k", 5)

    if not search_history and not viewed_product_ids and not current_search:
        return jsonify({"error": "Missing search input or viewed product ids"}), 400

    # ======== User Vector =========
    user_queries = []

    if search_history:
        user_queries.extend(search_history)

    if current_search:
        user_queries.append(current_search)

    if viewed_product_ids:
        for pid in viewed_product_ids:
            idx = product_id_to_index.get(pid)
            if idx is not None:
                user_queries.append(product_texts[idx])

    if not user_queries:
        return jsonify({"error": "No valid input for similarity calculation"}), 400

    # Transform user queries to TF-IDF vectors
    user_vectors = tfidf_vectorizer.transform(user_queries)
    
    # Calculate average similarity
    similarities = np.zeros(len(products))
    for user_vector in user_vectors:
        similarities += cosine_similarity(user_vector, tfidf_matrix).flatten()
    similarities /= len(user_queries)

    # ======== Filtering =========
    results = []
    for idx, p in enumerate(products):
        if not (min_price <= p["price"] <= max_price):
            continue
        if origin_filter and origin_filter.lower() not in p["origin"].lower():
            continue
        results.append((similarities[idx], p))

    # ======== Sort and return =========
    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    return jsonify({
        "recommended_products": [
            {
                "id": p["id"],
                "title": p["title"],
                "category": p["category"],
                "origin": p["origin"],
                "description": p["description"],
                "price": f"{p['price']:,}đ",
                "ocop_star": p["ocop_star"],
                "score": round(float(score), 4)
            } for score, p in results
        ]
    })

# ========== Similar Products API ==========
@app.route("/similar-products/<int:product_id>", methods=["GET"])
def similar_products(product_id):
    idx = product_id_to_index.get(product_id)
    if idx is None:
        return jsonify({"error": "Product not found"}), 404

    # Get the vector for the selected product
    product_vector = tfidf_matrix[idx]

    # Calculate similarity with all products
    similarities = cosine_similarity(product_vector, tfidf_matrix).flatten()

    # Get the top 5 similar products
    top_indices = similarities.argsort()[-6:-1][::-1]  # Exclude the product itself
    similar_products = [
        {
            "id": products[i]["id"],
            "title": products[i]["title"],
            "category": products[i]["category"],
            "origin": products[i]["origin"],
            "description": products[i]["description"],
            "price": f"{products[i]['price']:,}đ",
            "ocop_star": products[i]["ocop_star"],
            "score": round(float(similarities[i]), 4)
        } for i in top_indices
    ]

    return jsonify({
        "similar_products": similar_products
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
