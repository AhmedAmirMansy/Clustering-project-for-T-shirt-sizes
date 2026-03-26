#%matplotlib inline 
#to make the plots appear in the plot window in spyder


#%%





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time




from collections import Counter

def calculate_cluster_percentages(labels, total_samples):
    # Count the occurrences of each cluster label
    cluster_counts = Counter(labels)
    
    # Calculate the percentage of each cluster
    cluster_percentages = {label: (count / total_samples) * 100 for label, count in cluster_counts.items()}
    
    return cluster_percentages


# Step 1: Normalize the Data
def normalize_data(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)

# Step 2: Implement DBSCAN using functions
def region_query(X, point_idx, eps):
    #Compute the Euclidean distance between the given point and all other points in the dataset. Return indices of points within the radius eps
    """
    Find all points in the dataset within distance eps of point point_idx.
    """
    distances = np.linalg.norm(X - X[point_idx], axis=1)
    return np.where(distances < eps)[0]

def expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples):
    #convert all points in "neighbors" to cluster points of cluster with id "cluster_id"
    """
    Expand a new cluster from the initial point.
    """
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
       
        if labels[neighbor_idx] == -1:  # Unvisited point
            labels[neighbor_idx] = cluster_id
            new_neighbors = region_query(X, neighbor_idx, eps) #recursive, look for neighbors of neighbors in order to add them to the cluster
            if len(new_neighbors) >= min_samples:
                neighbors = np.concatenate((neighbors, new_neighbors))#this makes the loop does not stop untill all valid neighbors are counted
        if labels[neighbor_idx] == -1:  # Previously marked as noise
            labels[neighbor_idx] = cluster_id  # Convert noise to cluster point
        i += 1

def dbscan(X, eps, min_samples):
    """
    Perform DBSCAN clustering.
    """
    labels = -np.ones(X.shape[0], dtype=int)  # Initialize all points as noise (-1)
    cluster_id = 0

    for point_idx in range(X.shape[0]):
        if labels[point_idx] != -1:  # Already classified
            continue
        neighbors = region_query(X, point_idx, eps) #return the indices of neighbors in an array named neighbors
        if len(neighbors) < min_samples: #density requirements
            labels[point_idx] = -1  # Mark as noise
        else: #tenfa3 core point
            expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples)
            cluster_id += 1

    return labels

# Step 3: Visualize Clusters
def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    for label in unique_labels:
        class_member_mask = (labels == label)
        plt.scatter(X[class_member_mask, 0], X[class_member_mask, 1], label=f'Cluster {label}')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Step 4: Run DBSCAN
def run_dbscan(X, eps, min_samples):
    start_time = time.time()
    labels = dbscan(X, eps, min_samples)  # Use function-based DBSCAN
    elapsed_time = time.time() - start_time
    return labels, elapsed_time

# Step 5: Implement PCA
def pca(X, n_components=2): #this shall reduce the 5 parameters into 2 only
    mean = X.mean(axis=0)
    X_centered = X - mean
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    return X_centered @ eigenvectors[:, idx]

def main():
    sizes = [1000,2000,3000]
    times_no_pca = []
    times_with_pca = []
    eps_no_pca=[0.36,0.31,0.3]
    eps_pca=[0.3,0.244,0.228]
   
    for index, size in enumerate(sizes):
        alldata = pd.read_csv(r"C:\ath\guc\55-22567_ahmed_thabet_algorthisms project\DataSet.csv")
        data = alldata.values[:size]
        

        normalized_data = normalize_data(data)
        
    
        # Run DBSCAN on normalized data without PCA
            #eps is the min radius from the core point to another point for it to be a neighbor
            #min samples is the min number of neighbors for a point to be a core point
        labels_no_pca, _ = run_dbscan(normalized_data, eps=eps_no_pca[index], min_samples=1)
        
        # Calculate cluster percentages for no PCA
        cluster_percentages_no_pca = calculate_cluster_percentages(labels_no_pca, size)
        print(f"Cluster Percentages (Without PCA) for size {size}: {cluster_percentages_no_pca}")
        
        
        # Plot clusters without PCA
        plot_clusters(normalized_data, labels_no_pca, 'DBSCAN Clustering [5 Clusters] (Without PCA)')

        # Run DBSCAN without PCA
        _, elapsed_time_no_pca = run_dbscan(normalized_data, eps=eps_no_pca[index], min_samples=1)
        times_no_pca.append(elapsed_time_no_pca)
        
        # PCA
        reduced_data = pca(normalized_data)
        
        # Run DBSCAN on PCA-reduced data
        labels_with_pca, _ = run_dbscan(reduced_data, eps=eps_pca[index], min_samples=1)
        
        # Calculate cluster percentages for PCA
        cluster_percentages_with_pca = calculate_cluster_percentages(labels_with_pca, size)
        print(f"Cluster Percentages (With PCA) for size {size}: {cluster_percentages_with_pca}")


        # Plot clusters with PCA
        plot_clusters(reduced_data, labels_with_pca, 'DBSCAN Clustering [5 Clusters] (With PCA)')


        # Run DBSCAN with PCA
        _, elapsed_time_with_pca = run_dbscan(reduced_data, eps=eps_pca[index], min_samples=1)
        times_with_pca.append(elapsed_time_with_pca)
        
        
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_no_pca, marker='o', label='Without PCA')
    plt.plot(sizes, times_with_pca, marker='x', label='With PCA' )
    plt.title('DBSCAN Execution Time vs Sample Size')
    plt.xlabel('Sample Size')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(sizes)
    plt.legend()
    plt.grid(True)
    plt.show()

def main1():
    sizes = [1000,2000,3000]
    times_no_pca = []
    times_with_pca = [] 
    eps_no_pca=[0.38,0.323,0.308]
    eps_pca=[0.34,0.2479,0.231]  
    for index, size in enumerate(sizes):
        alldata = pd.read_csv(r"C:\ath\guc\55-22567_ahmed_thabet_algorthisms project\DataSet.csv")
        data = alldata.values[:size]

        normalized_data = normalize_data(data)
        
        # Run DBSCAN on normalized data without PCA
        labels_no_pca, _ = run_dbscan(normalized_data, eps=eps_no_pca[index], min_samples=1)
        
        # Calculate cluster percentages for no PCA
        cluster_percentages_no_pca = calculate_cluster_percentages(labels_no_pca, size)
        print(f"Cluster Percentages (Without PCA) for size {size}: {cluster_percentages_no_pca}")
        
        
        # Plot clusters without PCA
        plot_clusters(normalized_data, labels_no_pca, 'DBSCAN Clustering [3 Clusters] (Without PCA)')

        # Run DBSCAN without PCA
        _, elapsed_time_no_pca = run_dbscan(normalized_data, eps=eps_no_pca[index], min_samples=1)
        times_no_pca.append(elapsed_time_no_pca)
        
        # PCA
        reduced_data = pca(normalized_data)
        
        # Run DBSCAN on PCA-reduced data
        labels_with_pca, _ = run_dbscan(reduced_data, eps=eps_pca[index], min_samples=1)
        
        # Calculate cluster percentages for PCA
        cluster_percentages_with_pca = calculate_cluster_percentages(labels_with_pca, size)
        print(f"Cluster Percentages (With PCA) for size {size}: {cluster_percentages_with_pca}")


        # Plot clusters with PCA
        plot_clusters(reduced_data, labels_with_pca, 'DBSCAN Clustering [3 Clusters] (With PCA)')

        # Run DBSCAN with PCA
        _, elapsed_time_with_pca = run_dbscan(reduced_data, eps=eps_pca[index], min_samples=1)
        times_with_pca.append(elapsed_time_with_pca)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_no_pca, marker='o', label='Without PCA')
    plt.plot(sizes, times_with_pca, marker='x', label='With PCA' )
    plt.title('DBSCAN Execution Time vs Sample Size')
    plt.xlabel('Sample Size')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(sizes)
    plt.legend()
    plt.grid(True)
    plt.show()
    

if __name__  == "__main__":
    main()
    main1()







# %%
