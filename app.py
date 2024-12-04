from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

api_key = 'AIzaSyDa96Mfv5aK_quXMdK7OBWx4mztGMKvi-I'
cse_id = '941a1003b43d2486d'

app = Flask(__name__)

# Fungsi untuk mencari hasil di Google Custom Search
def google_search(query, api_key, cse_id, num_results=10):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
    search_results = []  # List untuk menyimpan hasil pencarian dalam format dictionary

    for item in res.get('items', []):  # Gunakan get untuk menghindari KeyError
        result = {
            'title': item.get('title', 'No title available'),
            'snippet': item.get('snippet', 'No snippet available'),
            'link': item.get('link', 'No link available')
        }
        search_results.append(result)

    return search_results

# Fungsi untuk menghitung cosine similarity
def calculate_cosine_similarity(query, search_results):
    # Gabungkan query dengan hasil pencarian
    all_texts = [query] + [result['title'] + " " + result['snippet'] for result in search_results]

    # Vectorisasi teks dengan TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Hitung cosine similarity antara query dan hasil pencarian
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_sim[0]

@app.route('/api/ai-recommendation', methods=['GET'])
def search():
    query = request.args.get('query', '')

    if not query:
        return jsonify({"error": "Query parameter is required!"}), 400

    search_results = google_search(query, api_key, cse_id, 10)

    if not search_results:
        return jsonify({"error": "No search results found!"}), 404

    cosine_sim = calculate_cosine_similarity(query, search_results)

    if not cosine_sim:
        return jsonify({"error": "Unable to calculate similarity!"}), 500

    result = {
        "code" : 200,
        "status" : "OK",
        "data": [
            {"index": idx+1, "similarity": round(cosine_sim[idx], 4), "title": search_results[idx]['title'], "snippet": search_results[idx]['snippet'], "link": search_results[idx]['link']}
            for idx in sorted(range(len(cosine_sim)), key=lambda i: cosine_sim[i], reverse=True)[:3]
        ]
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
