from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spectral image reconstruction and segmentation demo."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/test.jpg"),
        help="Path to the input image.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=32,
        help="Resize the image to size x size before processing.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=15,
        help="Number of nearest neighbors used for the pixel graph.",
    )
    parser.add_argument(
        "--embedding-k",
        type=int,
        default=10,
        help="Number of eigenvectors used to build the spectral embedding.",
    )
    parser.add_argument(
        "--reconstruction-k",
        type=int,
        default=50,
        help="Number of eigenvectors used for reconstruction.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of clusters used for spectral segmentation.",
    )
    parser.add_argument(
        "--spatial-weight",
        type=float,
        default=0.1,
        help="Weight for normalized pixel coordinates in the feature vector.",
    )
    parser.add_argument(
        "--color-weight",
        type=float,
        default=1.0,
        help="Weight for RGB values in the feature vector.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where generated figures will be saved.",
    )
    return parser.parse_args()


def load_image(image_path: Path, size: int) -> tuple[np.ndarray, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((size, size))
    image_array = np.asarray(resized, dtype=np.float64) / 255.0
    return image_array, resized


def build_features(
    image_array: np.ndarray,
    spatial_weight: float,
    color_weight: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    height, width, _ = image_array.shape
    pixels = image_array.reshape(-1, 3)

    ys, xs = np.mgrid[0:height, 0:width]
    coords = np.column_stack(
        [
            ys.ravel() / max(height - 1, 1),
            xs.ravel() / max(width - 1, 1),
        ]
    )

    features = np.hstack(
        [
            spatial_weight * coords,
            color_weight * pixels,
        ]
    )

    return features, pixels, height, width


def build_weight_matrix(features: np.ndarray, neighbors: int):
    pixel_count = features.shape[0]
    safe_neighbors = max(2, min(neighbors, pixel_count))

    model = NearestNeighbors(n_neighbors=safe_neighbors)
    model.fit(features)
    distances, indices = model.kneighbors(features)

    sigma = float(np.mean(distances))
    sigma = max(sigma, 1e-8)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for pixel_index in range(pixel_count):
        for neighbor_index, distance in zip(indices[pixel_index], distances[pixel_index]):
            rows.append(pixel_index)
            cols.append(int(neighbor_index))
            vals.append(float(np.exp(-(distance**2) / (2 * sigma**2))))

    weight_matrix = coo_matrix((vals, (rows, cols)), shape=(pixel_count, pixel_count))
    return (weight_matrix + weight_matrix.T) / 2


def compute_normalized_laplacian(weight_matrix):
    degrees = np.asarray(weight_matrix.sum(axis=1)).ravel()
    degree_matrix = diags(degrees)
    laplacian = degree_matrix - weight_matrix
    degree_inverse_sqrt = diags(1.0 / np.sqrt(degrees + 1e-8))
    normalized_laplacian = degree_inverse_sqrt @ laplacian @ degree_inverse_sqrt
    return normalized_laplacian


def spectral_decomposition(normalized_laplacian, eigen_count: int):
    matrix_size = normalized_laplacian.shape[0]
    safe_eigen_count = max(2, min(eigen_count, matrix_size - 1))
    eigenvalues, eigenvectors = eigsh(normalized_laplacian, k=safe_eigen_count, sigma=1e-5)
    return eigenvalues, eigenvectors


def reconstruct_image(pixels: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    coefficients = eigenvectors.T @ pixels
    reconstructed = eigenvectors @ coefficients
    return np.clip(reconstructed, 0.0, 1.0)


def segment_image(embedding: np.ndarray, clusters: int, height: int, width: int) -> np.ndarray:
    safe_clusters = max(2, min(clusters, embedding.shape[0]))
    labels = KMeans(n_clusters=safe_clusters, n_init=10, random_state=42).fit_predict(embedding)
    return labels.reshape(height, width)


def save_rgb_image(image: np.ndarray, destination: Path, title: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close()


def save_segment_image(segmented: np.ndarray, destination: Path, title: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(segmented, cmap="viridis")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_array, _ = load_image(args.image, args.size)
    features, pixels, height, width = build_features(
        image_array,
        spatial_weight=args.spatial_weight,
        color_weight=args.color_weight,
    )

    weight_matrix = build_weight_matrix(features, args.neighbors)
    normalized_laplacian = compute_normalized_laplacian(weight_matrix)

    _, embedding_eigenvectors = spectral_decomposition(
        normalized_laplacian,
        args.embedding_k,
    )
    embedding = embedding_eigenvectors[:, 1:min(4, embedding_eigenvectors.shape[1])]

    _, reconstruction_eigenvectors = spectral_decomposition(
        normalized_laplacian,
        args.reconstruction_k,
    )
    reconstructed_pixels = reconstruct_image(pixels, reconstruction_eigenvectors)
    reconstructed_image = reconstructed_pixels.reshape(height, width, 3)

    segmented = segment_image(embedding, args.clusters, height, width)

    save_rgb_image(image_array, args.output_dir / "original.png", "Original Image")
    save_rgb_image(
        reconstructed_image,
        args.output_dir / f"reconstruction_k{reconstruction_eigenvectors.shape[1]}.png",
        f"Spectral Reconstruction (k={reconstruction_eigenvectors.shape[1]})",
    )
    save_segment_image(
        segmented,
        args.output_dir / f"segmentation_c{np.unique(segmented).size}.png",
        f"Spectral Segmentation ({np.unique(segmented).size} clusters)",
    )

    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
