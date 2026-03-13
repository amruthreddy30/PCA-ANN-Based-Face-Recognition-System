import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import euclidean

class FaceRecognitionSystem:
    def __init__(self, dataset_path="dataset", img_size=(100, 100), k_eigenfaces=40, threshold=5000):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.k = k_eigenfaces
        self.threshold = threshold
        self.label_dict = {}
        self.mean_face = None
        self.eigenfaces = None
        self.model = None
        self.X_train_centered = None
        self.y_train = None
        self.X_train_proj = None

    def load_dataset(self):
        images, labels = [], []
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            print(f"Created empty dataset directory at '{self.dataset_path}'. Please add subfolders with images.")
            return np.array([]), np.array([])
            
        for label_id, person in enumerate(sorted(os.listdir(self.dataset_path))):
            person_path = os.path.join(self.dataset_path, person)
            if not os.path.isdir(person_path): continue
            
            self.label_dict[label_id] = person
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(cv2.resize(img, self.img_size).flatten())
                    labels.append(label_id)
        
        return np.array(images), np.array(labels)

    def train_pca(self, X_train):
        self.mean_face = np.mean(X_train, axis=0)
        self.X_train_centered = X_train - self.mean_face
        
        # Surrogate covariance matrix trick for huge matrices
        cov_matrix = np.dot(self.X_train_centered, self.X_train_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigen components in descending order
        idx = np.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        
        # Compute real eigenfaces and normalize
        self.eigenfaces = np.dot(self.X_train_centered.T, eigenvectors)
        for i in range(self.eigenfaces.shape[1]):
            self.eigenfaces[:, i] /= np.linalg.norm(self.eigenfaces[:, i])

    def train_ann(self, y_train):
        self.y_train = y_train
        eigenfaces_k = self.eigenfaces[:, :self.k]
        self.X_train_proj = np.dot(self.X_train_centered, eigenfaces_k)
        
        self.model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        self.model.fit(self.X_train_proj, self.y_train)

    def evaluate(self, X_test, y_test, max_k=80):
        print("Evaluating effect of k (Number of Eigenfaces)...")
        X_test_centered = X_test - self.mean_face
        k_vals = list(range(10, min(max_k + 1, self.eigenfaces.shape[1] + 1), 10))
        accuracies = []
        
        for k in k_vals:
            ek = self.eigenfaces[:, :k]
            model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
            model.fit(np.dot(self.X_train_centered, ek), self.y_train)
            acc = accuracy_score(y_test, model.predict(np.dot(X_test_centered, ek)))
            accuracies.append(acc)
            print(f"k = {k:02d} | Accuracy = {acc:.2f}")
            
        plt.plot(k_vals, accuracies, marker='o', linestyle='-')
        plt.title("Accuracy vs Number of Eigenfaces (k)")
        plt.xlabel("k (Eigenfaces)")
        plt.ylabel("Accuracy")
        plt.grid()
        if os.path.exists("results"):
            plt.savefig("results/accuracy_graph.png")
        else:
            plt.show()
        print("Evaluation graph generated.")

    def recognize(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return "Image not found."
        
        img_centered = cv2.resize(img, self.img_size).flatten() - self.mean_face
        img_proj = np.dot(img_centered, self.eigenfaces[:, :self.k])
        
        # Imposter Detection Check
        distances = [euclidean(img_proj, t) for t in self.X_train_proj]
        if min(distances) > self.threshold:
            return "Unknown Person (Imposter Detected!)"
            
        prediction = self.model.predict([img_proj])[0]
        return self.label_dict[prediction]

if __name__ == "__main__":
    if not os.path.exists("results"): os.makedirs("results")
    
    frs = FaceRecognitionSystem()
    X, y = frs.load_dataset()
    
    if len(X) == 0:
        print("No image data found. Please place images in the 'dataset' subfolders.")
    else:
        # Step 4: Train-Test Split (60% training - 40% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        
        print(f"Dataset Loaded. Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
        
        # Step 5-8: Train PCA
        frs.train_pca(X_train)
        
        # Step 9: Train ANN Model
        frs.train_ann(y_train)
        
        # Step 10 & 11: Evaluate and Graph Accuracy vs k
        frs.evaluate(X_test, y_test)
        
        # Step 12 & 13: Optional prediction run if a test.jpg is placed
        if os.path.exists("test.jpg"):
            print("Prediction for test.jpg:", frs.recognize("test.jpg"))
