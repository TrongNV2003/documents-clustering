import os
import glob
import umap
import torch
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from clustering_module.utils import constant
from clustering_module.services.embedding_models import Embedding
from clustering_module.services.document_extractor import Extractor
from clustering_module.preprocessing.text_preprocessing import Preprocessing
from clustering_module.visualize.visualizer import plot_dendrogram
from clustering_module.preprocessing.deduplicate import filter_duplicate_documents


def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer

def get_model(checkpoint: str, device: torch.device) -> AutoModel:
    model = AutoModel.from_pretrained(checkpoint).to(device)
    return model


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

parser = argparse.ArgumentParser(description="Document Clustering")
parser.add_argument("--model", type=str, default="models/vietnamese-sbert", help="Model checkpoint", required=True)
parser.add_argument("--documents_dir", type=str, default="documents", help="Directory containing documents", required=True)
parser.add_argument("--is_deduplicate", action="store_true", default=False, help="Enable deduplication")
parser.add_argument("--max_clusters", type=int, default=15, help="Max of clusters for Elbow Method")
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
    extractor = Extractor()
    
    file_patterns = [os.path.join(args.documents_dir, "*.docx"), os.path.join(args.documents_dir, "*.pdf")]
    documents = []
    for pattern in file_patterns:
        documents.extend(glob.glob(pattern))
    documents = sorted(documents)
    
    if len(documents) < 2:
        print("Required at least 2 documents to cluster.")
        return

    doc_embeddings, document_names = [], []
    
    logger.info("Đang embed tài liệu...")
    for path in tqdm(documents, desc="Embedding docs"):
        try:
            raw = extractor.extract_document(path)
            if not raw.strip():
                print(f"Skipped document: '{os.path.basename(path)}' due to extraction failure.")
                continue
            
            preprocessed_doc = preprocessor.preprocess(raw)
            vec = embedding_model.embed_document(preprocessed_doc)
            
            if vec is None or not np.any(vec):
                print(f"Skipped document: '{os.path.basename(path)}': Failed to generate embedding.")
                continue
            doc_embeddings.append(vec)
            document_names.append(os.path.basename(path))
        
        except Exception as e:
            print(f"Error processing {os.path.basename(path)}: {e}")
            continue

    if len(doc_embeddings) < 1:
        print("Không có tài liệu hợp lệ để xử lý.")
        return

    if args.is_deduplicate:
        print("Bắt đầu loại bỏ tài liệu trùng lặp...")
        doc_embeddings, document_names = filter_duplicate_documents(
            doc_embeddings, document_names, constant.DUPLICATE_THRESHOLD
        )

    X = np.stack(doc_embeddings)
    
    n_components = min(len(doc_embeddings) - 1, 100)
    pca = PCA(n_components=n_components, random_state=42)
    X = pca.fit_transform(X)
    
    if len(X) < 2:
        print("Không đủ tài liệu để phân cụm. Cần ít nhất 2 tài liệu.")
        return
    
    if len(document_names) >= 2:
        print("\nCosine Similarity giữa các tài liệu:")
        similarities = cosine_similarity(X)
        for i in range(len(document_names)):
            for j in range(i + 1, len(document_names)):
                print(f"{document_names[i]} vs {document_names[j]}: {similarities[i][j]:.4f}")
                if similarities[i][j] > constant.SIMILARITY_THRESHOLD:
                    print(f"→ Có thể cùng thể loại (similarity cao)")
                else:
                    print(f"→ Có thể khác thể loại (similarity thấp)")
    else:
        print("Không đủ tài liệu để tính toán cosine similarity.")

    n_clusters = find_optimal_k(X, args.max_clusters)

    logger.info(f"Clustering params: n_clusters={n_clusters}")

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average',
        metric='cosine'
    )
    
    labels = clustering.fit_predict(X)
    for name, lbl in zip(document_names, labels):
        print(f"{name} → Cluster {lbl}")

    plot_dendrogram(X, document_names, constant.DISTANCE_THRESHOLD)

    
if __name__ == "__main__":
    main()