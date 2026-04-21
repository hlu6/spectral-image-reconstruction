# Spectral Image Reconstruction

This mini project explores how an image can be represented as a graph and analyzed with spectral methods.

## What the project does

- Loads an input image and resizes it to a small working resolution
- Builds a pixel graph using spatial position and RGB color
- Computes the normalized graph Laplacian
- Reconstructs the image from a limited number of Laplacian eigenvectors
- Segments the image using spectral embedding and k-means clustering

## Why it is interesting

Instead of looking at an image as just a grid of pixels, this project builds a graph where similar pixels are connected. From there, the Laplacian eigenvectors are used in two ways: to map pixels into a feature space for segmentation, and to rebuild the image using smooth graph-based frequency patterns.

## Project structure

```text
spectral-image-reconstruction/
|- data/
|  |- test.jpg
|- outputs/
|  |- original.png
|  |- reconstruction_k50.png
|  |- segmentation_c5.png
|- notebooks/
|  |- spectral_demo.ipynb
|- main.py
|- requirements.txt
|- README.md
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

You can also change parameters:

```bash
python main.py --size 64 --neighbors 20 --reconstruction-k 75 --clusters 6
```

## Main ideas in the code

- Pixel features combine normalized coordinates and RGB values
- A k-nearest-neighbor graph connects similar pixels
- The normalized graph Laplacian captures image structure
- Eigenvectors of the Laplacian provide a spectral basis for reconstruction
- K-means on the spectral embedding produces a simple segmentation map

## Output

Running the script saves images to the `outputs/` folder so the results are easy to include in a GitHub showcase.

## Notebook demo

For a more visual walkthrough, open `notebooks/spectral_demo.ipynb`. It shows the graph setup, reconstruction with different values of `k`, and the segmentation step in a more explanation-friendly format.

## Next improvements

- Compare several reconstruction values of `k`
- Measure how graph parameters affect quality
- Add side-by-side plots directly to the README
