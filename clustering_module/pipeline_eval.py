import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from kneed import KneeLocator
from typing import List, Tuple
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)

from clustering_module.utils import constant
from clustering_module.services.embedding_models import Embedding
from clustering_module.preprocessing.text_preprocessing import Preprocessing
from clustering_module.preprocessing.deduplicate import filter_duplicate_documents
from clustering_module.visualize.visualizer import plot_dendrogram, visualize_clusters


def find_optimal_k(X, max_clusters=20):
    """
    Tìm số cụm bằng Elbow Method.
    """
    sse = []
    n_samples = len(X)
    max_clusters = min(max_clusters, n_samples)

    k_values = range(1, max_clusters + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    kneedle = KneeLocator(k_values, sse, curve="convex", direction="decreasing")
    optimal_k = kneedle.knee

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, sse, 'bx-')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f"Optimal K = {optimal_k}")
    plt.xlabel('Số cụm (K)')
    plt.ylabel('Tổng bình phương sai số (SSE)')
    plt.title('Elbow Method để tìm K')
    plt.legend()
    plt.show()

    print(f"Số cụm tối ưu dựa trên Elbow Method: {optimal_k}")
    return optimal_k

def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer

def get_model(checkpoint: str, device: torch.device) -> AutoModel:
    model = AutoModel.from_pretrained(checkpoint).to(device)
    return model

def load_dataset(dataset_path: str) -> Tuple[List[str], List[str], List[int]]:
    df = pd.read_csv(dataset_path)
    texts = df['text'].tolist()
    document_names = [f"doc_{id}" for id in df['id']]
    label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
    labels = [label_mapping[label] for label in df['label']]
    return texts, document_names, labels

def evaluate_clustering(true_labels: List[int], predicted_labels: List[int]) -> dict:
    results = {}
    results['ARI'] = adjusted_rand_score(true_labels, predicted_labels)
    results['NMI'] = normalized_mutual_info_score(true_labels, predicted_labels)
    results["FMI"] = fowlkes_mallows_score(true_labels, predicted_labels)
    results["Homogeneity"] = homogeneity_score(true_labels, predicted_labels)
    results["Completeness"] = completeness_score(true_labels, predicted_labels)
    results["V-measure"] = v_measure_score(true_labels, predicted_labels)
    return results

parser = argparse.ArgumentParser(description="Document Clustering")
parser.add_argument("--model", type=str, default="models/vietnamese-sbert", help="Model checkpoint")
parser.add_argument("--dataset", type=str, help="Path to CSV dataset (optional)")
args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(args.model)
    model = get_model(args.model, device)
    embedding_model = Embedding(
        model=model,
        tokenizer=tokenizer,
        device=device,
        pooling_type="weighted",
    )
    preprocessor = Preprocessing(
        force_text=True,
        segment=True
    )
    texts, document_names, true_labels = load_dataset(args.dataset)
    
    unique_labels = set(true_labels)
    logger.info(f"\nCó {len(unique_labels)} nhãn khác nhau trong dữ liệu.")
    
    if len(texts) < 1:
        print("Không có tài liệu hợp lệ để xử lý.")
        return

    doc_embeddings = []
    valid_indices = []
    
    logger.info("Đang embed tài liệu...")
    for idx, text in enumerate(tqdm(texts, desc="Generating embeddings")):
        try:
            text = preprocessor.preprocess(text)
            vec = embedding_model.embed_document(text)
            if vec is None or not np.any(vec):
                print(f"Skipped document: '{document_names[idx]}': Failed to generate embedding.")
                continue
            doc_embeddings.append(vec)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error embedding {document_names[idx]}: {e}")
            continue

    if len(doc_embeddings) < 1:
        print("Không có tài liệu hợp lệ để xử lý.")
        return

    document_names = [document_names[i] for i in valid_indices]
    true_labels = [true_labels[i] for i in valid_indices]

    doc_embeddings, document_names = filter_duplicate_documents(
        doc_embeddings, document_names, constant.DUPLICATE_THRESHOLD
    )
    
    if args.dataset:
        valid_indices_after_filter = []
        for name in document_names:
            idx = [i for i, n in enumerate([f"doc_{id}" for id in pd.read_csv(args.dataset)['id']]) if n == name][0]
            valid_indices_after_filter.append(idx)
        true_labels = [true_labels[i] for i in valid_indices_after_filter]

    if len(doc_embeddings) < 2:
        print(f"Chỉ còn {len(doc_embeddings)} tài liệu hợp lệ sau khi lọc trùng lặp. Cần ít nhất 2 tài liệu để tính cosine similarity và phân cụm.")
        print(f"Tài liệu còn lại: {document_names[0] if document_names else 'Không có'}")
        return

    X = np.stack(doc_embeddings)
    
    n_components = min(len(doc_embeddings) - 1, 200)
    pca = PCA(n_components=n_components, random_state=42)
    X = pca.fit_transform(X)
    
    max_clusters = 20
    n_clusters = find_optimal_k(X, max_clusters)
    
    logger.info(f"Number of clusters found: {n_clusters}")
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average',
        metric='cosine'
    )
    
    labels = clustering.fit_predict(X)
    print(f"Số cụm được tìm thấy: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"Số tài liệu không thuộc cụm nào: {list(labels).count(-1)}")

    eval_results = evaluate_clustering(true_labels, labels)
    print(f"Adjusted Rand Index (ARI): {eval_results['ARI']:.4f}")
    print(f"Normalized Mutual Information (NMI): {eval_results['NMI']:.4f}")
    print(f"Fowlkes-Mallows Index (FMI): {eval_results['FMI']:.4f}")
    print(f"Homogeneity: {eval_results['Homogeneity']:.4f}")
    print(f"Completeness: {eval_results['Completeness']:.4f}")
    print(f"V-measure: {eval_results['V-measure']:.4f}")

    visualize_clusters(X, labels, true_labels, document_names, "Document Clusters")
    plot_dendrogram(X, document_names, constant.DISTANCE_THRESHOLD)

if __name__ == "__main__":
    main()
