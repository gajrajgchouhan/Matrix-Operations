import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from collections import Counter

image_dir = "lfwcrop_grey/faces"
celebrity_photos = os.listdir(image_dir)
celebrity_images = np.array([os.path.join(image_dir, photo) for photo in celebrity_photos])
celebrity_names = np.array([name[: name.find("0") - 1].replace("_", " ") for name in celebrity_photos])

allowed = [name for name,freq in Counter(celebrity_names).items() if freq>=70]
allowed_index = np.array([i for i in range(len(celebrity_names)) if celebrity_names[i] in allowed])

celebrity_images = celebrity_images[allowed_index]
celebrity_names = celebrity_names[allowed_index]

print("Loading images...")

arrayimages = np.array([cv2.imread(image, 0).ravel() for image in celebrity_images], dtype=np.float64).T

print("Splitting using train_test_split")

X_train, X_test, y_train, y_test = train_test_split(range(arrayimages.shape[1]), celebrity_names, test_size=0.2)

X_train = arrayimages[:,X_train]
X_test = arrayimages[:,X_test]

no_of_train = X_train.shape[1]
no_of_test = X_test.shape[1]

mean = np.mean(X_train, axis=1, keepdims=True)
X_train -= mean
X_test -= mean

print("Finding covariance and eigens")

cov = X_train.T @ X_train
cov = cov / (cov.shape[0] - 1)

# eigenvalue of covariance is === variance
eigval, eigvec = np.linalg.eig(cov)

order = eigval.argsort()[::-1]
eigval = eigval[order]
eigvec = eigvec[:, order]
# sorted the eigvec according to eigval

K = int(input("Input K (default 250) : ") or 250)
u = X_train @ eigvec[:, :K]
u = u / np.linalg.norm(u, axis=0) # eigenvectors are unit vector

reconstructed = (X_train.T @ u)
test_constructed = (X_test.T @ u)

# train x test
er = np.zeros((no_of_train, no_of_test), dtype="float")
for i in range(no_of_train):
    for j in range(no_of_test):
        er[i,j] = np.linalg.norm(reconstructed[i,:] - test_constructed[j,:])

pred = y_train[np.argmin(er, axis=0)]
print(accuracy_score(y_test, pred) * 100, "% accuracy")
print(classification_report(y_test, pred))
