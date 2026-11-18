# Iris K-Means Clustering

A simple machine learning project that applies **K-Means clustering** to
the Iris dataset using Python and scikit-learn. The project groups the
flowers into **three clusters** and visualizes the results with two 2D
scatter plots created using Seaborn.

## Project Overview

The Iris dataset contains measurements for 150 flowers across three
species:

-   Setosa
-   Versicolor
-   Virginica

Each flower is described by four numerical features:

  Feature          Description
  ---------------- -----------------------------
  `sepal.length`   Sepal length in centimeters
  `sepal.width`    Sepal width in centimeters
  `petal.length`   Petal length in centimeters
  `petal.width`    Petal width in centimeters
  `variety`        Actual flower species

K-Means is an **unsupervised learning algorithm**. It does not use the
`variety` column when learning. Instead, it groups samples according to
similarities in their four measurements.

## How It Works

1.  **Load the data:** Read `iris.csv` with Pandas.
2.  **Select the features:** Use the four numerical measurements as
    input to the model.
3.  **Standardize the features:** Apply `StandardScaler` so that each
    feature contributes on a comparable scale.
4.  **Train K-Means:** Fit `KMeans` with `n_clusters=3`,
    `init="k-means++"`, `n_init=10`, and `random_state=42`.
5.  **Get cluster assignments:** Retrieve the predicted cluster for each
    flower.
6.  **Visualize the results:** Create separate Sepal and Petal scatter
    plots and show the three cluster centroids on each plot.

### Why standardize the data?

The four measurements have different ranges. Standardization transforms
each feature to have approximately zero mean and unit variance. This
prevents features with larger numerical scales from dominating the
distance calculations used by K-Means.

### What are the centroids?

A centroid is the center of a cluster, calculated as the mean position
of the samples assigned to that cluster. The plots display the three
centroids as large red **X** markers.

The model learns in standardized four-dimensional feature space. For
visualization, the centroids are transformed back to the original
centimeter scale, then projected onto the two features shown in each
plot.

## Visualizations

### 1. Sepal Features

This scatter plot displays **Sepal Length** against **Sepal Width**.
Each point represents one flower, and its color indicates the cluster
assigned by K-Means. The red X markers show the three cluster centroids
projected onto the sepal dimensions.

![K-Means clustering using sepal length and sepal
width](https://github.com/user-attachments/assets/490c548e-8d6a-4112-94ad-87682652e22b)

### 2. Petal Features

This scatter plot displays **Petal Length** against **Petal Width**. As
in the first plot, point colors represent the K-Means cluster
assignments and the red X markers represent the projected centroids.

![K-Means clustering using petal length and petal
width](https://github.com/user-attachments/assets/e4ca9adc-4746-4d88-90a7-0427105fb753)

> **Note:** The two plots are different 2D views of the same clustering
> model. K-Means is trained using all four features, not separately on
> the two feature pairs. Therefore, the colors show the original
> four-feature cluster assignments projected onto each 2D plot.

## Requirements

-   Python 3.8+
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   scikit-learn

Install the dependencies with:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Run the Project

1.  Clone the repository:

    ``` bash
    git clone https://github.com/YOUR-USERNAME/iris_k-means.git
    ```

2.  Open the project directory:

    ``` bash
    cd iris_k-means
    ```

3.  Make sure `iris.csv` is in the project directory.

4.  Run the Python script:

    ``` bash
    python main.py
    ```

## Understanding the Results

-   **Cluster labels (`0`, `1`, and `2`)** are identifiers assigned by
    K-Means. They do not inherently correspond to Setosa, Versicolor, or
    Virginica.
-   **Inertia** is the sum of squared distances from each sample to its
    assigned cluster center. Lower inertia indicates tighter clusters,
    but it should not be used alone to judge clustering quality.
-   **Cluster distribution** reports how many samples were assigned to
    each cluster.
-   **Centroids** indicate the center of each learned cluster.

Because the actual species labels are not used during training, they can
optionally be compared with the predicted clusters afterward to explore
how well the unsupervised groups align with the known Iris species.

## Technologies Used

-   **Python** --- programming language
-   **Pandas** --- loading and handling the CSV dataset
-   **NumPy** --- numerical operations
-   **scikit-learn** --- feature standardization and K-Means clustering
-   **Matplotlib** --- plot creation
-   **Seaborn** --- statistical scatter plot styling

------------------------------------------------------------------------

*This project is an educational implementation of K-Means clustering on
the Iris dataset.*


