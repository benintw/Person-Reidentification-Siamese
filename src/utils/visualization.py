"""This module provides visualization utilities for person re-identification:

- plot_closest_imgs: Plot closest images
- plot_training_history: Plot training history
- visualize_embeddings: Visualize embeddings
- visualize_query_results: Visualize query results

"""

__all__ = [
    "plot_closest_imgs",
    "plot_training_history",
    "visualize_embeddings",
    "visualize_query_results",
]

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from skimage import io
from sklearn.manifold import TSNE


def plot_closest_imgs(
    anc_img_names, DATA_DIR, image, img_path, closest_idx, distance, no_of_closest=10
):
    """Plot closest images in a network graph visualization.

    Args:
        anc_img_names: DataFrame column or Series containing gallery image paths
        DATA_DIR: Directory containing the images
        image: Optional pre-loaded image (not used if img_path is provided)
        img_path: Path to the query image
        closest_idx: Indices of closest gallery images
        distance: Distance values for each gallery image
        no_of_closest: Number of closest matches to display
    """
    import os
    from pathlib import Path

    # Create a graph
    G = nx.Graph()

    # Extract the filename from the path
    if isinstance(img_path, str):
        query_filename = img_path.split("/")[-1]
    else:
        query_filename = str(img_path)

    # Prepare image names list starting with query image
    S_name = [query_filename]

    # Add gallery image names
    for s in range(min(no_of_closest, len(closest_idx))):
        S_name.append(anc_img_names.iloc[closest_idx[s]])

    # Convert DATA_DIR to Path object if it's not already
    if not isinstance(DATA_DIR, Path):
        DATA_DIR = Path(DATA_DIR)

    # Create a placeholder colored image for missing files
    def create_placeholder(index, label):
        # Create a colored placeholder based on index
        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "yellow",
        ]
        color = colors[index % len(colors)]

        # Create a simple colored image with text
        fig = plt.figure(figsize=(3, 3))
        plt.fill([0, 1, 1, 0], [0, 0, 1, 1], color)
        plt.text(0.5, 0.5, label, ha="center", va="center", color="white", fontsize=12)
        plt.axis("off")

        # Convert figure to image array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img

    # Load images or create placeholders
    for i in range(len(S_name)):
        try:
            # Try to load the image
            img_path = DATA_DIR / S_name[i]
            if os.path.exists(img_path):
                image = io.imread(str(img_path))
            else:
                # If file doesn't exist, create a placeholder
                label = f"Image {i}\n{S_name[i][:10]}..."
                image = create_placeholder(i, label)
                print(f"Warning: Image not found: {img_path}, using placeholder")
        except Exception as e:
            # If loading fails, create a placeholder
            label = f"Image {i}\n{S_name[i][:10]}..."
            image = create_placeholder(i, label)
            print(f"Error loading image {i}: {e}")

        # Add node to graph with the image
        G.add_node(i, image=image)

    # Add edges from query to gallery images
    for j in range(1, min(no_of_closest + 1, len(S_name))):
        G.add_edge(0, j, weight=distance[closest_idx[j - 1]])

    # Create layout
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception as e:
        print(f"Error creating layout: {e}, falling back to spring layout")
        pos = nx.spring_layout(G)

    # Create figure
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111)
    ax.set_aspect("equal")
    nx.draw_networkx_edges(G, pos, ax=ax)

    # Set limits for the plot
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    # Get transformation functions
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform

    # Set image size
    piesize = 0.1  # this is the image size
    p2 = piesize / 2.0

    # Add images to the nodes
    for n in G:
        try:
            # Get coordinates
            xx, yy = trans(pos[n])  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates

            # Create subplot for the image
            a = plt.axes([xa - p2, ya - p2, piesize, piesize])
            a.set_aspect("equal")

            # Display the image
            a.imshow(G.nodes[n]["image"])

            # Add a title (first 10 chars of filename)
            if len(S_name[n]) > 10:
                title = S_name[n][0:10] + "..."
            else:
                title = S_name[n]

            # Add distance info for gallery images
            if n > 0 and n - 1 < len(closest_idx):
                dist_val = distance[closest_idx[n - 1]]
                title += f"\n{dist_val:.2f}"

            a.set_title(title, fontsize=8)
            a.axis("off")
        except Exception as e:
            print(f"Error adding image {n} to plot: {e}")

    # Hide the main axes
    ax.axis("off")

    # Add a title to the figure
    plt.suptitle(f"Query: {S_name[0]} and Top {no_of_closest} Matches", fontsize=16)

    # Save the figure before showing it
    try:
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/query_results.png", bbox_inches="tight")
        print("Saved visualization to results/query_results.png")
    except Exception as e:
        print(f"Error saving figure: {e}")

    # Display the plot
    plt.show()


def visualize_query_results(predictions, query_idx, data_dir, no_of_closest=10):
    """Visualize query results using plot_closest_imgs.

    Args:
        predictions: Output from predictor.predict()
        query_idx: Index of the query to visualize
        data_dir: Directory containing the images
        no_of_closest: Number of closest matches to display
    """
    try:
        # Ensure data_dir is a proper path
        from pathlib import Path

        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"Warning: Data directory {data_dir} does not exist")

        # Get data for the specified query
        query_path = predictions["query_img_paths"][query_idx]
        gallery_paths = predictions["gallery_img_paths"]
        rankings = predictions["rankings"][query_idx]
        distances = predictions["distances"][query_idx]

        # Limit the number of closest matches to the available gallery size
        no_of_closest = min(no_of_closest, len(gallery_paths))

        # Get the closest matches
        closest_idx = rankings[:no_of_closest]

        # Verify that the query image exists
        query_full_path = data_dir / query_path
        if not query_full_path.exists():
            print(f"Warning: Query image {query_full_path} does not exist")
            print(f"Trying relative path: {query_path}")
            # Try using the path as is
            query_full_path = query_path

        # Create a DataFrame with gallery image paths
        import pandas as pd

        gallery_df = pd.DataFrame({"path": gallery_paths})

        print(f"Visualizing results for query {query_idx}: {query_path}")
        print(f"Top {no_of_closest} matches:")
        for i, idx in enumerate(closest_idx):
            print(f"  {i+1}. {gallery_paths[idx]} (distance: {distances[idx]:.4f})")

        # Call the plot function
        plot_closest_imgs(
            anc_img_names=gallery_df["path"],
            DATA_DIR=data_dir,
            image=None,  # Not needed as we're passing the path directly
            img_path=query_path,
            closest_idx=closest_idx,
            distance=distances,
            no_of_closest=no_of_closest,
        )
    except Exception as e:
        print(f"Error visualizing query results: {e}")
        import traceback

        traceback.print_exc()


def plot_training_history(history: dict, save_path: Path | str | None = None) -> None:
    """Plot training history metrics including loss, mAP, and CMC scores.

    Args:
        history: Dictionary containing training metrics per epoch
        save_path: Optional path to save the plot image
    """
    # Determine which metrics are available in the history
    has_loss = "train_loss" in history and "val_loss" in history
    has_map = "mAP" in history and len(history["mAP"]) > 0
    has_cmc = "rank1" in history and len(history["rank1"]) > 0

    # Create appropriate number of subplots based on available metrics
    num_plots = sum([has_loss, has_map, has_cmc])
    if num_plots == 0:
        print("No metrics to plot")
        return

    fig = plt.figure(figsize=(6 * num_plots, 5))

    plot_idx = 1

    # Plot Loss History
    if has_loss:
        ax1 = fig.add_subplot(1, num_plots, plot_idx)
        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["val_loss"], label="Validation")
        ax1.set_title("Loss History")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        plot_idx += 1

    # Plot mAP History
    if has_map:
        ax2 = fig.add_subplot(1, num_plots, plot_idx)
        ax2.plot(history["mAP"], label="mAP", color="green")
        ax2.set_title("Mean Average Precision History")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mAP")
        ax2.legend()
        ax2.grid(True)
        plot_idx += 1

    # Plot CMC Scores History
    if has_cmc:
        ax3 = fig.add_subplot(1, num_plots, plot_idx)
        if "rank1" in history:
            ax3.plot(history["rank1"], label="Rank-1", color="blue")
        if "rank5" in history:
            ax3.plot(history["rank5"], label="Rank-5", color="orange")
        if "rank10" in history:
            ax3.plot(history["rank10"], label="Rank-10", color="red")
        ax3.set_title("CMC Scores History")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Score")
        ax3.legend()
        ax3.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_embeddings(model, dataloader, device):

    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            anchor, positive, negative, person_id = batch
            anchor = anchor.to(device)
            emb = model(anchor).cpu().numpy()
            embeddings.extend(emb)
            labels.extend(person_id)

    embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        idx = [l == label for l in labels]
        # Convert label to string to ensure we can slice it
        label_str = str(label)
        # Use the first 4 characters of the label, or the whole label if it's shorter
        label_display = label_str[: min(4, len(label_str))]
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=label_display, alpha=0.6)

    plt.legend()
    plt.title("t-SNE Visualization of Embeddings")

    # Create logs directory if it doesn't exist
    import os

    os.makedirs("logs", exist_ok=True)

    # Save figure first, then show it
    plt.savefig("logs/embeddings.png")
    plt.show()


def visualize_embedding_space(predictions, model, combined_loader, device):
    """Visualize the embedding space using t-SNE.

    Args:
        predictions: Output from predictor.predict()
        model: The trained model
        combined_loader: DataLoader containing both query and gallery images
        device: Device to run inference on
    """
    # Simply call your existing function
    visualize_embeddings(model, combined_loader, device)
