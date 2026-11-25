import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps=eps
        self.min_samples=min_samples
        self.labels_=None

    def region_query(self, X, point_idx):
        distances=np.linalg.norm(X-X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def fit(self,X):
        n=len(X)
        labels=np.full(n,-2)

        cluster_id=0

        for px in range(n):
            if labels[px]!=-2:
                continue

            neighbors=self.region_query(X, px)

            if len(neighbors)<self.min_samples:
                labels[px]=-1
                continue

            labels[px]=cluster_id

            i=0
            while i<len(neighbors):
                neighbor_idx=neighbors[i]

                if labels[neighbor_idx]==-1:
                    labels[neighbor_idx]=cluster_id

                if labels[neighbor_idx]!=-2:
                    i+=1
                    continue

                labels[neighbor_idx]=cluster_id

                new_neighbors=self.region_query(X, neighbor_idx)
                if len(new_neighbors)>=self.min_samples:
                    neighbors=np.concatenate((neighbors,new_neighbors))

                i+=1

            cluster_id+=1
        self.labels_=labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_