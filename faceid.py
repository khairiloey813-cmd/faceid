import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

def charger_et_separer_donnees(ratio_train=0.8):
    """Charge Olivetti et sépare les poses par individu (ex: 8 train, 2 test)."""
    dataset = fetch_olivetti_faces()
    images = dataset.images
    labels = dataset.target
    N_total = len(labels)
    
    # 40 sujets, 10 poses chacun
    indices_train = []
    indices_test = []
    
    for personne in range(40):
        idx = np.where(labels == personne)[0]
        n_train = int(len(idx) * ratio_train)
        indices_train.extend(idx[:n_train])
        indices_test.extend(idx[n_train:])
        
    X_train = images[indices_train].reshape(len(indices_train), -1).T
    y_train = labels[indices_train]
    
    X_test = images[indices_test].reshape(len(indices_test), -1).T
    y_test = labels[indices_test]
    
    return X_train, y_train, X_test, y_test

def entrainer_pca_gram(X_train, k=40):
    """Calcule le visage moyen et les k meilleures Eigenfaces via Turk & Pentland."""
    x_bar = np.mean(X_train, axis=1, keepdims=True)
    X_c = X_train - x_bar
    
    # Matrice de Gram (N_train x N_train)
    S = np.dot(X_c.T, X_c)
    valeurs_propres, V_gram = np.linalg.eigh(S)
    
    # Tri décroissant et troncature
    idx = np.argsort(valeurs_propres)[::-1]
    V_k = V_gram[:, idx[:k]]
    
    # Projection dans l'espace image et orthonormalisation
    U_k = np.dot(X_c, V_k)
    U_k = U_k / np.linalg.norm(U_k, axis=0)
    
    # Signatures d'apprentissage : projection des visages de train
    # Dimensions : (k, N_train)
    signatures_train = np.dot(U_k.T, X_c)
    
    return U_k, x_bar, signatures_train

def reconnaitre_visage(x_inconnu, U_k, x_bar, signatures_train, y_train):
    """Projette une nouvelle image et trouve l'identité la plus proche (1-NN)."""
    phi = x_inconnu - x_bar
    w = np.dot(U_k.T, phi)
    
    # Calcul des distances euclidiennes avec toutes les signatures mémorisées
    distances = np.linalg.norm(signatures_train - w, axis=0)
    index_plus_proche = np.argmin(distances)
    
    prediction = y_train[index_plus_proche]
    distance_min = distances[index_plus_proche]
    
    return prediction, distance_min

if __name__ == "__main__":
    print("1. Préparation du jeu d'entraînement et de test...")
    X_train, y_train, X_test, y_test = charger_et_separer_donnees(ratio_train=0.8)
    print(f"   Train set : {X_train.shape[1]} visages | Test set : {X_test.shape[1]} visages")
    
    k = 40
    print(f"2. Entraînement de l'espace spectral (k = {k})...")
    U_k, x_bar, signatures_train = entrainer_pca_gram(X_train, k=k)
    
    print("3. Évaluation sur l'ensemble de test...")
    succes = 0
    total = X_test.shape[1]
    
    for i in range(total):
        img_test = X_test[:, i:i+1]
        label_reel = y_test[i]
        label_predit, dist = reconnaitre_visage(img_test, U_k, x_bar, signatures_train, y_train)
        
        if label_predit == label_reel:
            succes += 1
            
    precision = (succes / total) * 100
    print(f"\n==========================================")
    print(f"   Résultat global : {succes}/{total} visages reconnus")
    print(f"   Taux de reconnaissance (Accuracy) : {precision:.2f} %")
    print(f"==========================================")