from flask import Flask, request, jsonify
import numpy as np
import pymysql
import os
import json
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread, Lock
import redis

load_dotenv()
app = Flask(__name__)

# ========== Config ==========

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
}

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# ========== Globals ==========

products = []
product_texts = []
product_id_to_index = {}
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = None
product_lock = Lock()

# ========== Load Product Functions ==========

def load_products_from_mysql():
    try:
        connection = pymysql.connect(**MYSQL_CONFIG)
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            query = """
                SELECT 
                    p.id, 
                    p.name, 
                    p.category, 
                    s.province, 
                    p.description, 
                    p.price, 
                    p.star,
                    p.status
                FROM product p
                JOIN seller s ON p.sellerId = s.id
            """
            cursor.execute(query)
            results = cursor.fetchall()
        connection.close()
        print("✅ Loaded products from MySQL")
        return results
    except Exception as e:
        print(f"⚠️ MySQL error: {e}")
        return None

# ========== Rebuild TF-IDF ==========

def rebuild_tfidf():
    global product_texts, product_id_to_index, tfidf_matrix
    print("rebuild vcl")
    with product_lock:
        product_texts = []
        product_id_to_index.clear()

        for idx, p in enumerate(products):
            full_text = f"{p['name']} {p['category']} {p['province']} {p['description']}"
            product_texts.append(full_text)
            product_id_to_index[p['id']] = idx
        print("before fit transform")
        tfidf_matrix = tfidf_vectorizer.fit_transform(product_texts)
        print("✅ TF-IDF rebuilt")

# ========== Redis CRUD Handling ==========

def cache_product(product):
    global products
    for i, p in enumerate(products):
        if p["id"] == product["id"]:
            products[i] = product
            rebuild_tfidf()
            print(f"� Updated product {product['id']}")
            return
    products.append(product)
    rebuild_tfidf()
    print(f"� Added new product {product['id']}")

def delete_product(product_id):
    global products
    products = [p for p in products if p["id"] != product_id]
    
    rebuild_tfidf()

def listen_to_redis():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe("product-events")

    print("� Listening to Redis on channel 'product-events'...")
    for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            data = json.loads(message["data"])
            event = data.get("event")
            payload = data.get("data")

            if event == "create" or event == "update":
                cache_product(payload)
            elif event == "delete":
                delete_product(data["id"])
                
        except Exception as e:
            print(f"❌ Error handling Redis event: {e}")

# ========== Initial Load ==========

products = load_products_from_mysql()

if not products:
    raise Exception("❌ No product data found")

rebuild_tfidf()

# ========== Flask APIs ==========

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    search_history = data.get("search_history", [])
    viewed_product_ids = data.get("viewed_product_ids", [])
    current_search = data.get("current_search", "")
    province_filter = data.get("province")
    min_price = data.get("min_price", 0)
    max_price = data.get("max_price", float("inf"))
    top_k = data.get("top_k", 5)

    if not search_history and not viewed_product_ids and not current_search:
        return jsonify({"error": "Missing search input or viewed product ids"}), 400

    with product_lock:
        if not products or tfidf_matrix is None or tfidf_matrix.shape[0] != len(products):
            return jsonify({"error": "Product data is being updated. Please retry shortly."}), 503

        product_texts_copy = list(product_texts)
        tfidf_matrix_copy = tfidf_matrix.copy()
        product_id_map_copy = dict(product_id_to_index)
        products_copy = list(products)

    user_queries = []

    if search_history:
        user_queries.extend(search_history)

    if current_search:
        user_queries.append(current_search)

    for pid in viewed_product_ids:
        idx = product_id_map_copy.get(pid)
        if idx is not None and idx < len(product_texts_copy):
            user_queries.append(product_texts_copy[idx])

    if not user_queries:
        return jsonify({"error": "No valid input for similarity calculation"}), 400

    try:
        user_vectors = tfidf_vectorizer.transform(user_queries)
        similarities = np.zeros(len(products_copy))
        for user_vector in user_vectors:
            similarities += cosine_similarity(user_vector, tfidf_matrix_copy).flatten()
        similarities /= len(user_queries)
    except Exception as e:
        return jsonify({"error": f"Error during similarity calculation: {str(e)}"}), 500

    results = []
    for idx, p in enumerate(products_copy):
        if not (min_price <= p["price"] <= max_price):
            continue
        if province_filter and province_filter.lower() not in p["province"].lower():
            continue
        results.append((similarities[idx], p))

    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    return jsonify({
        "recommended_products": [
            {
                "id": p["id"],
                "name": p["name"],
                "category": p["category"],
                "province": p["province"],
                "description": p["description"],
                "price": f"{p['price']:,}đ",
                "star": p["star"],
                "score": round(float(score), 4)
            } for score, p in results
        ]
    })

@app.route("/similar-products/<int:product_id>", methods=["GET"])
def similar_products(product_id):
    with product_lock:
        idx = product_id_to_index.get(product_id)
        if idx is None or idx >= len(products):
            return jsonify({"error": "Product not found"}), 404

        product_vector = tfidf_matrix[idx]
        similarities = cosine_similarity(product_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-6:-1][::-1]

        result = []
        for i in top_indices:
            product = products[i]
            result.append({
                "id": product["id"],
                "name": product["name"],
                "category": product["category"],
                "province": product["province"],
                "description": product["description"],
                "price": f"{product['price']:,}đ",
                "star": product["star"],
                "score": round(float(similarities[i]), 4)
            })

    return jsonify({"similar_products": result})

# ========== Main ==========

if __name__ == "__main__":
    redis_thread = Thread(target=listen_to_redis, daemon=True)
    redis_thread.start()
    app.run(host="0.0.0.0", port=5000)
