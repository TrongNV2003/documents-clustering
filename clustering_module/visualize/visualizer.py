import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from loguru import logger
from typing import List
import umap
import pandas as pd

def plot_dendrogram(X, document_names, distance_threshold):
    """
    Vẽ dendrogram cho Hierarchical Clustering.
    X: Ma trận embeddings.
    document_names: Danh sách tên tài liệu.
    distance_threshold: Ngưỡng khoảng cách để cắt cây.
    """
    Z = linkage(X, method='average', metric='cosine')
    
    plt.figure(figsize=(8, 6))
    dendrogram(
        Z,
        labels=document_names,
        distance_sort='descending',
        color_threshold=distance_threshold,
        above_threshold_color='gray',
        leaf_rotation=45,
        leaf_font_size=10
    )
    plt.axhline(y=distance_threshold, color='r', linestyle='--', label=f'Threshold ({distance_threshold})')
    plt.title('Dendrogram cho Hierarchical Clustering')
    plt.xlabel('Tài liệu')
    plt.ylabel('Cosine Distance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dendrogram.png')
    plt.show()
    
def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, true_labels: np.ndarray, document_names: List[str], title: str):
    """
    Trực quan hóa các cụm trong không gian 2D bằng UMAP.
    """
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    logger.info("Reduced embeddings to 2D for visualization")

    labels_str = [f"Cluster {lbl}" for lbl in labels]
    true_labels_str = [f"True {lbl}" for lbl in true_labels]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab20', s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"{title} (Predicted Clusters)")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.savefig("clusters_2d.png")
    plt.close()
    logger.info("Saved Matplotlib scatter plot to clusters_2d.png")

    df_plot = pd.DataFrame({
        'UMAP1': embeddings_2d[:, 0],
        'UMAP2': embeddings_2d[:, 1],
        'Cluster': labels_str,
        'True Label': true_labels_str,
        'Document': document_names
    })
    fig = px.scatter(
        df_plot,
        x='UMAP1',
        y='UMAP2',
        color='Cluster',
        symbol='True Label',
        hover_data=['Document', 'True Label'],
        title=f"{title} (Predicted Clusters with True Labels)"
    )
    fig.update_layout(width=1000, height=800)
    pio.write_html(fig, file="clusters_2d.html")
    logger.info("Saved Plotly scatter plot to clusters_2d.html")