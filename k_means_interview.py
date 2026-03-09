"""
K-Means Clustering — Manual Implementation in NumPy
Algorithm:
  - K-Means clustering
  - Euclidean distance metric
  - Iterative centroid updates until convergence
Rules:
  - Do NOT use sklearn.cluster.KMeans or any clustering library
  - Do NOT use scipy.cluster or similar clustering functions
  - Do NOT use any prebuilt distance functions (e.g., scipy.spatial.distance)
Allowed:
  - numpy 
  - Prefer vectorized NumPy implementations over explicit nested loops.

Target:
  - Successfully cluster the synthetic blob dataset
  - Achieve inertia (within-cluster sum of squares) that decreases monotonically
  - Final clustering should match ground truth labels with ≥90% accuracy (up to permutation)
"""

import numpy as np
np.random.seed(42)

# ---------------------------------
# Data: Synthetic 2D Gaussian Blobs
# ---------------------------------
def make_blobs(n_samples=300, n_clusters=3, cluster_std=1.2):
    """Generate synthetic clustered data with overlapping clusters."""
    # Closer centers with higher variance = more realistic overlapping clusters
    centers = np.array([
        [0.0, 0.0],
        [3.0, 3.0],
        [-2.5, 3.5],
    ])[:n_clusters]
    
    # Varying cluster sizes (more realistic)
    cluster_sizes = [n_samples // 2, n_samples // 3, n_samples - n_samples // 2 - n_samples // 3]
    
    X_list, y_list = [], []
    
    for i, (center, size) in enumerate(zip(centers, cluster_sizes)):
        # Different spread per cluster (anisotropic)
        std = cluster_std * (1.0 + 0.3 * i)  # Slightly different std per cluster
        X_cluster = center + std * np.random.randn(size, 2)
        X_list.append(X_cluster)
        y_list.append(np.full(size, i))
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # Shuffle the data
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

X, y_true = make_blobs(n_samples=300, n_clusters=3, cluster_std=1.2)

# ---------------------------------
# Hyperparameters
# ---------------------------------
n_clusters = 3
max_iters = 100
tol = 1e-4  # Convergence tolerance

# ---------------------------------
# K-Means Implementation
# ---------------------------------



def compute_cluster(means, X):
    """
    a point belong to the cluster of a mean if the point is closest to the mean
    X:[n, 2]
    means: [k, 2]

    output: labels[n,1]


    compute distance 
    ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c.T
    for i in range(len(X)):
        diff_to_means = means - X[i,:]  #(k,2)
        distance = np.linalg.norm(diff_to_means, axis=1) #(k)
        label_x = np.argmin(distance)
    assign to the smallest disance    
    """
    label_x = np.zeros((len(X),))
    for i in range(len(X)):
        diff_to_means = means - X[i,:]  #(k,2)
        distance = np.linalg.norm(diff_to_means, axis=1) #(k,)
        label_x[i] = np.argmin(distance)
    
    return label_x


def compute_means(X,label_x,k):
    """
    new_means = np.zeros((k,2))
    for i range(k):
        cluster = X[label_x==i] #(n,2)
        new_means[i] = np.mean(cluster, axis = 0)  #(1,2)
    """
    new_means = np.zeros((k,2)) 
    cluster_distance_sum = np.zeros((k,))
    largest_cluster_idx = np.argmax([np.sum(label_x==i) for i in range(k)])
    largest_cluster = X[label_x==largest_cluster_idx] # (size_large, 2)
    for i in range(k):
        cluster = X[label_x==i] #(n,2)
        if len(cluster)==0:
            mean_idx = np.random.choice(np.arange(len(largest_cluster))) # 1
            new_means[i] = largest_cluster[mean_idx] #(1, 2)
        else:
            new_means[i] = np.mean(cluster, axis = 0)  #(1,2)
        cluster_distance_sum[i] = np.sum(np.linalg.norm(cluster-new_means[i], axis=1))
    return new_means, np.sum(cluster_distance_sum)

def check_empty_clusters(label_x):
    """
    len(label_x[i])==0
    """
    empty_cluster_idx = []
    for i in range(k):
        if len(label_x[i])==0:
            empty_cluster_idx.append(i)
    return empty_cluster_idx
def sample_from_large_cluster(label_x, X, num_empty,k):
    """
    find the largest_cluster: 
        idx = np.argmax([label_x[i] for i in range(k)])
        cluster = X[label_x==idx]

    mean_idx = np.random.choice(np.arange(len(cluster)),num_empty)
    centroids = X[mean_idx]

    """
    idx = np.argmax([np.sum(int(label_x==i)) for i in range(k)])
    cluster = X[label_x==idx]
    mean_idx = np.random.choice(np.arange(len(cluster)),num_empty)
    centroids = X[mean_idx]
    return centroids

def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    Run K-Means clustering algorithm.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Convergence tolerance for centroid movement
    
    Returns:
        centroids: Final centroid positions (k, n_features)
        labels: Final cluster assignments (n_samples,)
        inertia_history: List of inertia values per iteration
    
    len(X)>K
    If len(X)==k:
        return X
    If  len(X)< k:
        return

    initialize the means (randomly selects points from X)
    loop over update steps
        compute the clustering based on the means
        [follow-up] if there are n empty clusters:
                sample n means from the largest clusters
        compute new means
        break if the new mean == old means
        

    """
    # TODO: Implement the full K-Means algorithm
    # kmeans++
    mean_idx = np.random.choice(np.arange(len(X)),k, replace=False)
    centroids = X[mean_idx]
    inertia_history = []
    for i in range(max_iters):
        label_x = compute_cluster(centroids, X)
        # check_empty_clusters(label_x)

        new_centroids,  inertia = compute_means(X,label_x,k)
        inertia_history.append(inertia)
        if np.linalg.norm(new_centroids-centroids)<tol:
            break
        centroids = new_centroids
        
    return centroids, label_x, inertia_history
    



# ---------------------------------
# Evaluation Utilities
# ---------------------------------

def clustering_accuracy(y_true, y_pred):
    """
    Compute clustering accuracy accounting for label permutation.
    Since cluster labels are arbitrary, we find the best mapping.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
    
    Returns:
        accuracy: Best accuracy under any label permutation
    """
    from itertools import permutations
    
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    k = len(unique_true)
    
    best_acc = 0.0
    for perm in permutations(range(k)):
        # Map predicted labels according to permutation
        y_mapped = np.zeros_like(y_pred)
        for i, p in enumerate(perm):
            y_mapped[y_pred == i] = p
        acc = np.mean(y_mapped == y_true)
        best_acc = max(best_acc, acc)
    
    return best_acc


# ---------------------------------
# Run
# ---------------------------------
if __name__ == "__main__":
    print(f"Data shape: {X.shape}")
    print(f"Number of clusters: {n_clusters}")
    print("-" * 50)
    
    # Run K-Means
    centroids, labels, inertia_history = kmeans(X, n_clusters, max_iters=max_iters, tol=tol)
    
    # Print inertia progression
    print("Inertia progression:")
    for i, inertia in enumerate(inertia_history):
        if i < 5 or i == len(inertia_history) - 1:
            print(f"  Iteration {i+1:3d}: {inertia:.4f}")
        elif i == 5:
            print("  ...")
    
    # Check monotonic decrease
    is_monotonic = all(inertia_history[i] >= inertia_history[i+1] 
                       for i in range(len(inertia_history)-1))
    print(f"\nInertia monotonically decreasing: {is_monotonic}")
    
    # Compute clustering accuracy
    acc = clustering_accuracy(y_true, labels)
    print(f"Clustering accuracy (vs ground truth): {acc:.3f}")
    
    # Final summary
    print("-" * 50)
    print(f"Final inertia: {inertia_history[-1]:.4f}")
    print(f"Converged in {len(inertia_history)} iterations")
    print(f"Final centroids:\n{centroids}")
