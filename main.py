import numpy as np

from algorithm import DBSCAN

def main():
    X = np.array([[1.0,1.0],[2.1,3.7],[4.2,5.3],[6.7,8.2],[5.0,5.0],[8.4,5.7],[9.2,6.4],[10.0,10.0],[3.3,4.5],[9.6,7.8]])
    eps=1.5
    min_samples=3
    model=DBSCAN(eps=eps,min_samples=min_samples)
    labels=model.fit_predict(X)
    print("Results:")
    for i,label in enumerate(labels):
        print(f"Point {X[i]} is in cluster {label}")

    cluster_labels = set(labels) - {-1}
    print(f"Number of clusters found: {len(cluster_labels)}")
    print(f"Number of noise points: {list(labels).count(-1)}")

if __name__ == "__main__":
    main()

