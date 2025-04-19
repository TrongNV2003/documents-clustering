import os
import glob
import umap
import torch
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
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

parser = argparse.ArgumentParser(description="Document Clustering")
parser.add_argument("--model", type=str, default="models/vietnamese-sbert", help="Model checkpoint", required=True)
parser.add_argument("--documents_dir", type=str, default="documents", help="Directory containing documents", required=True)
parser.add_argument("--is_deduplicate", action="store_true", default=False, help="Enable deduplication")
parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters (optional)")

args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(args.model)
    model = get_model(args.model, device)
    embedding_model = Embedding(
        model=model,
        tokenizer=tokenizer,
        device=device,
        pooling_type="cls", 
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

    doc_embeddings = []
    document_names = []
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
    
    # reducing dimensionality
    n_neighbors = min(5, len(doc_embeddings) - 1)
    print(n_neighbors)
    
    # X = umap.UMAP(
    #     n_components=3,
    #     n_neighbors=n_neighbors,
    #     random_state=42,
    # ).fit_transform(X)
    
    # Reducing dimensionality using PCA
    X = PCA(
        n_components=None,
        svd_solver='full',
        random_state=42
    ).fit_transform(X)
    
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

    n_clusters = args.n_clusters
    distance_threshold = constant.DISTANCE_THRESHOLD
    if n_clusters is not None:
        distance_threshold = None
    else:
        n_clusters = None
    logger.info(f"Clustering params: n_clusters={n_clusters}, distance_threshold={distance_threshold}")

    print("\nKết quả phân cụm bằng Hierarchical Clustering:")
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage='average',
        metric='cosine'
    )
    labels = clustering.fit_predict(X)
    for name, lbl in zip(document_names, labels):
        print(f"{name} → Cluster {lbl}")

    # Vẽ dendrogram
    plot_dendrogram(X, document_names, constant.DISTANCE_THRESHOLD)
    

if __name__ == "__main__":
    main()