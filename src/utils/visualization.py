"""
Embedding visualization using UMAP and t-SNE.
For latent space analysis and error analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. Install with: pip install umap-learn")


def extract_embeddings(model, dataloader, device="cpu"):
    """
    Extract fused embeddings from the model.

    Returns:
        embeddings: numpy array of shape [n_samples, embedding_dim]
        labels: numpy array of shape [n_samples]
    """
    model.eval()
    model = model.to(device)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for visual, text, labels in tqdm(dataloader, desc="Extracting embeddings"):
            visual = visual.to(device)
            text = text.to(device)

            # Get embeddings before classification layer
            embeddings = model.get_embeddings(visual, text)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_embeddings), np.concatenate(all_labels)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    save_path: str,
    perplexity: int = 30,
    title: str = "t-SNE Visualization of Learned Embeddings",
):
    """
    Create t-SNE visualization of embeddings.
    """
    print(f"Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab20",
        alpha=0.6,
        s=10,
    )

    # Add legend with class names
    unique_labels = np.unique(labels)
    handles = [
        plt.scatter(
            [],
            [],
            c=[plt.cm.tab20(label / len(unique_labels))],
            label=class_names[label],
            s=50,
        )
        for label in unique_labels[:20]  # Limit legend to 20 classes
    ]
    plt.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
    )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved to: {save_path}")


def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    save_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "UMAP Visualization of Learned Embeddings",
):
    """
    Create UMAP visualization of embeddings.
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available, skipping...")
        return

    print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab20",
        alpha=0.6,
        s=10,
    )

    # Add legend
    unique_labels = np.unique(labels)
    handles = [
        plt.scatter(
            [],
            [],
            c=[plt.cm.tab20(label / len(unique_labels))],
            label=class_names[label],
            s=50,
        )
        for label in unique_labels[:20]
    ]
    plt.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
    )

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP plot saved to: {save_path}")


def plot_embedding_comparison(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    save_dir: str,
):
    """
    Create both t-SNE and UMAP visualizations.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # t-SNE
    plot_tsne(
        embeddings,
        labels,
        class_names,
        save_path=save_dir / "tsne_embeddings.png",
    )

    # UMAP
    if UMAP_AVAILABLE:
        plot_umap(
            embeddings,
            labels,
            class_names,
            save_path=save_dir / "umap_embeddings.png",
        )


if __name__ == "__main__":
    # Test with random data
    print("Testing visualization with random data...")

    n_samples = 500
    embedding_dim = 128
    n_classes = 10

    embeddings = np.random.randn(n_samples, embedding_dim)
    labels = np.random.randint(0, n_classes, n_samples)
    class_names = [f"Class_{i}" for i in range(n_classes)]

    plot_embedding_comparison(
        embeddings, labels, class_names, save_dir="outputs/figures/test_viz"
    )
    print(" Visualization test complete!")
