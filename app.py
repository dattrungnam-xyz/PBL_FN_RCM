from flask import Flask, request, jsonify
import numpy as np
import pymysql
import os
import json
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread, Lock
from collections import namedtuple
import redis
import re

# Load .env
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

ProductData = namedtuple(
    "ProductData", ["products", "product_texts", "product_id_to_index", "tfidf_matrix"]
)
shared_data = None
data_lock = Lock()
tfidf_vectorizer = TfidfVectorizer()

# ========== Helper functions ==========


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text


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


def rebuild_tfidf(new_products):
    product_texts = []
    product_id_to_index = {}

    for idx, p in enumerate(new_products):
        full_text = f"{p['name']} {p['category']} {p['province']} {p['description']}"
        processed_text = preprocess_text(full_text)
        product_texts.append(processed_text)
        product_id_to_index[p["id"]] = idx  # ✅ FIXED: Gán ID → index

    tfidf_matrix = tfidf_vectorizer.fit_transform(product_texts)

    new_data = ProductData(
        products=new_products,
        product_texts=product_texts,
        product_id_to_index=product_id_to_index,
        tfidf_matrix=tfidf_matrix,
    )

    with data_lock:
        global shared_data
        shared_data = new_data

    print("✅ TF-IDF rebuilt & updated")


# ========== Redis Event Handling ==========


def cache_product(product):
    with data_lock:
        current_products = list(shared_data.products) if shared_data else []

    for i, p in enumerate(current_products):
        if p["id"] == product["id"]:
            current_products[i] = product
            break
    else:
        current_products.append(product)

    rebuild_tfidf(current_products)


def delete_product(product_id):
    with data_lock:
        current_products = list(shared_data.products) if shared_data else []

    new_products = [p for p in current_products if p["id"] != product_id]
    rebuild_tfidf(new_products)


def listen_to_redis():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe("product-events")

    print("📡 Listening to Redis channel 'product-events'...")
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
            print(f"❌ Redis event error: {e}")


# ========== Initial Load ==========

initial_products = load_products_from_mysql()
if not initial_products:
    raise Exception("❌ No product data found")

rebuild_tfidf(initial_products)

# ========== Flask APIs ==========


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    search_history = data.get("search_history", [])
    viewed_product_ids = data.get("viewed_product_ids", [])
    categories = data.get("categories")
    provinces_filter = data.get("provinces", [])
    min_price = data.get("min_price", 0)
    max_price = data.get("max_price", float("inf"))
    page = int(data.get("page", 1))
    page_size = int(data.get("page_size", 15))
    search = data.get("search", "").lower().strip()

    with data_lock:
        data_snapshot = shared_data

    if not data_snapshot:
        return (
            jsonify({"error": "Product data is being updated. Please retry shortly."}),
            503,
        )

    similarities = np.zeros(len(data_snapshot.products))
    total_weight = 0

    for query in search_history:
        user_vector = tfidf_vectorizer.transform([query])
        similarities += (
            1.0 * cosine_similarity(user_vector, data_snapshot.tfidf_matrix).flatten()
        )
        total_weight += 1.0

    for pid in viewed_product_ids:
        idx = data_snapshot.product_id_to_index.get(pid)
        if idx is not None and idx < len(data_snapshot.product_texts):
            user_vector = tfidf_vectorizer.transform([data_snapshot.product_texts[idx]])
            similarities += (
                0.5
                * cosine_similarity(user_vector, data_snapshot.tfidf_matrix).flatten()
            )
            total_weight += 0.5

    if total_weight > 0:
        similarities /= total_weight
    else:
        similarities = np.ones(len(data_snapshot.products))

    results = []
    for idx, p in enumerate(data_snapshot.products):
        if max_price == float("inf"):
            if not (int(min_price) <= int(p["price"])):
                continue
        else:
            if not (int(min_price) <= int(p["price"]) <= int(max_price)):
                continue
        if provinces_filter and not any(
            province.lower() in p["province"].lower() for province in provinces_filter
        ):
            continue
        if categories and not any(
            category.lower() in p["category"].lower() for category in categories
        ):
            continue
        if (
            search
            and search not in p["name"].lower()
            and search not in p.get("description", "").lower()
        ):
            continue
        if p["status"] == "STOPPED":
            continue
        results.append((similarities[idx], p))

    results = sorted(results, key=lambda x: x[0], reverse=True)
    total_results = len(results)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_results = results[start:end]

    return jsonify(
        {
            "total": total_results,
            "page": page,
            "page_size": page_size,
            "recommended_products": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "category": p["category"],
                    "province": p["province"],
                    "description": p["description"],
                    "price": f"{p['price']:,}đ",
                    "star": p["star"],
                    "score": round(float(score), 4),
                }
                for score, p in paginated_results
            ],
        }
    )


@app.route("/similar-products/<product_id>", methods=["GET"])
def similar_products(product_id):
    with data_lock:
        data_snapshot = shared_data

    if not data_snapshot:
        return jsonify({"error": "Product data is not ready"}), 503

    # Ensure product_id matches type in map
    try:
        if isinstance(next(iter(data_snapshot.product_id_to_index)), int):
            product_id = int(product_id)
        else:
            product_id = str(product_id)
    except Exception as e:
        return jsonify({"error": "Invalid product_id"}), 400

    idx = data_snapshot.product_id_to_index.get(product_id)
    if idx is None or idx >= len(data_snapshot.products):
        return jsonify({"error": "Product not found"}), 404

    product_vector = data_snapshot.tfidf_matrix[idx]
    similarities = cosine_similarity(
        product_vector, data_snapshot.tfidf_matrix
    ).flatten()
    top_indices = similarities.argsort()[-50:-1][::-1]

    result = []
    for i in top_indices:
        product = data_snapshot.products[i]
        if product["status"] == "STOPPED":
            continue
        if len(result) >= 10:
            break
        result.append(
            {
                "id": product["id"],
                "name": product["name"],
                "category": product["category"],
                "province": product["province"],
                "description": product["description"],
                "price": f"{product['price']:,}đ",
                "star": product["star"],
                "score": round(float(similarities[i]), 4),
            }
        )

    return jsonify({"similar_products": result})


# ========== Main ==========

if __name__ == "__main__":
    redis_thread = Thread(target=listen_to_redis, daemon=True)
    redis_thread.start()
    app.run(host="0.0.0.0", port=5000)
