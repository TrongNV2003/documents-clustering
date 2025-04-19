from sklearn.metrics.pairwise import cosine_similarity

def filter_duplicate_documents(embeddings, names, similarity_threshold=0.95):
    """
    Lọc các tài liệu trùng lặp dựa trên cosine similarity.
    Giữ lại tài liệu đầu tiên trong cặp có similarity >= threshold.
    """
    if len(embeddings) < 2:
        return embeddings, names
    
    similarities = cosine_similarity(embeddings)
    keep_indices = list(range(len(embeddings)))
    to_remove = set()
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarities[i][j] >= similarity_threshold and i not in to_remove:
                if names[j].endswith('.pdf'):
                    print(f"Duplicate detected: {names[i]} (similarity {similarities[i][j]:.4f} with {names[j]}). Keeping {names[j]}.")
                    to_remove.add(i)
                else:
                    print(f"Duplicate detected: {names[j]} (similarity {similarities[i][j]:.4f} with {names[i]}). Keeping {names[i]}.")
                    to_remove.add(j)
    
    keep_indices = [i for i in range(len(embeddings)) if i not in to_remove]
    filtered_embeddings = [embeddings[i] for i in keep_indices]
    filtered_names = [names[i] for i in keep_indices]
    
    return filtered_embeddings, filtered_names