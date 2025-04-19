# Clustering Documents
This project implements a pipeline for clustering long documents in the Vietnamese language. Using Agglomerative Hierarchical Clustering algorithm to cluster documents based on their semantic similarity, embeddings by Sentence-Bert [Vietnamese-SBert](https://huggingface.co/keepitreal/vietnamese-sbert) . 

The pipeline includes document extraction, text preprocessing, embedding generation, duplicate filtering, clustering, and visualization via dendrograms.

Key features:
- Extracts text from .docx and .pdf files.
- Preprocesses Vietnamese text (e.g., chunking, segmentation).
- Reducing dimensionality using PCA or UMAP (768 -> 100 dimensions).
- Filters out duplicate documents (cosine similarity ≥ 0.95).
- Visualizes clustering results with a dendrogram.

## Installation
``sh
pip install -r requirements.txt
``

## Usage
First, put your documents into "documents" folder, then run:
``sh
bash run.sh
``

## Dataset
Dataset: [hiuman/vietnamese_classification](https://huggingface.co/datasets/hiuman/vietnamese_classification)


## Output evaluation
Evaluate model from test set of Dataset
1. Metrics
- Adjusted Rand Index (ARI): Đo lường sự tương đồng giữa các cụm do mô hình tạo ra và các lớp thực tế
- Normalized Mutual Information (NMI): Đo lường sự phụ thuộc lẫn nhau giữa phân cụm và nhãn thực tế bằng cách sử dụng lý thuyết thông tin
- Fowlkes-Mallows Index (FMI): Trung bình hình học của precision và recall khi xem xét các cặp điểm thuộc cùng một cụm và cùng một lớp
- Homogeneity: Đo lường xem mỗi cụm chứa chủ yếu các điểm dữ liệu thuộc cùng một lớp hay không
- Completeness: Đo lường xem tất cả các điểm dữ liệu thuộc cùng một lớp có được gán cho cùng một cụm hay không.
- V-measure: Là trung bình điều hòa của Homogeneity và Completeness

2. Results
| Models                | ARI       | NMI       | FMI       | V-measure |
|---------------------- |:---------:|:---------:|:---------:|:---------:|
| Our method + PCA      | 40.43     | 63.95     | 44.40     | 63.95     |
| Our method + UMAP     | 44.72     | 66.21     | 48.08     | 66.21     |