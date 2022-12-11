import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 

class Metric:

    @classmethod
    def confusion_matrix(self,true,pred):
        unique_classes = np.unique(true)
        n_classes = len(unique_classes)

        counts = np.array([np.sum(true == unique_classes[i]) for i in range(n_classes)],dtype=int)
        cm = np.zeros((n_classes,n_classes),dtype=int)
        # confusion matrix
        for i in range(n_classes):
            for j in range(n_classes):
                cm[i,j] = np.sum(pred[true == unique_classes[i]]==unique_classes[j])

        accuracy = np.diag(cm) / counts

        recall = np.diag(cm) / np.sum(cm, axis = 1)

        precision = np.diag(cm) / np.sum(cm, axis = 0)

        """
        Precision = TP / TP + FP (the ability of the classifier not to label as positive a sample that is negative)
        Recall = TP / TP + FN (the ability of the classifier to find all the positive samples)
        """

        f1 = 2 / (1/recall + 1/precision)

        total_accuracy = np.sum(true==pred) / true.size

        with np.printoptions(precision=2, suppress=True, floatmode="fixed"):

            s = "\n\n".join((
                f"Total accuray {total_accuracy:.02f}",
                f"Classes => {str(unique_classes)}",
                f"Confusion Matrix ",
                str(cm),
                f"Accuracy (in %) {accuracy*100}",
                f"Precision (in %) {precision*100}",
                f"Recall (in %) {recall*100}",
                f"F1 (in %) {f1*100}"))

        return s, total_accuracy, accuracy, precision, recall, f1

class PCA:
    def __init__(self,k) -> None:
        self.k = k

    def fit(self,A,y=None):
        A = np.asarray(A).astype("float")

        self.mean = np.mean(A, axis=0)
        A -= self.mean
        
        cov = A.T @ A / (A.shape[0] - 1)
        eigval, self.eigvec = np.linalg.eigh(cov)
        order = eigval.argsort()[::-1]
        
        self.eigvec = self.eigvec[:, order][:, :self.k]
        return self

    def transform(self,A):
        A = np.asarray(A).astype("float")
        A -= self.mean
        return (A @ self.eigvec)

class LDA:
    def __init__(self,k) -> None:
        self.k = k 

    def fit(self,X,y):

        X = np.asarray(X).astype(float)
        self.overall_mean = np.mean(X,axis=0,keepdims=True)
        X -= self.overall_mean 

        unique_classes = np.unique(y)
        n_features = X.shape[1]

        S_w = np.zeros((n_features, n_features))
        class_mean = []
        class_data_size = []

        for u in unique_classes:
            i_th_class = X[y == u,:]
            m = np.mean(i_th_class, axis=0, keepdims=True)
            i_th_class -= m
            S_w += (i_th_class.T @ i_th_class)  
            class_mean.append(m)
            class_data_size.append(i_th_class.shape[0])

        S_b = np.zeros_like(S_w)
        for mean, N in zip(class_mean, class_data_size):
            S_b += N * ((self.overall_mean - mean) @ (self.overall_mean - mean).T)

        S = np.linalg.inv(S_w) @ S_b

        eigval, self.eigvec = np.linalg.eigh(S)
        order = eigval.argsort()[::-1]

        self.eigvec = self.eigvec[:, order][:, :self.k]
        return self

    def transform(self,X):
        X = np.asarray(X).astype(float)
        X -= self.overall_mean
        return (X @ self.eigvec)

def plot_classification_region(model, x, y, filename, title=""):

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, ax = plt.subplots()

    Z = model.predict(np.stack([xx.ravel(), yy.ravel()], axis=1))
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.7)
    ax.scatter(x[:, 0], x[:, 1], c=y, s=25, edgecolor="k")
    ax.set_title(title)
    f.savefig(f"Region for {filename}.png")
    plt.close()

def preprocess_data(cls,x_train,x_test,y_train,k):
    transformer = cls(k).fit(x_train,y_train)
    x_train_transformed = transformer.transform(x_train)
    x_test_transformed = transformer.transform(x_test)
    return x_train_transformed, x_test_transformed

def add_key(m, k, v):
    if k not in m:
        m[k] = []
    m[k].append(v)
    return m 

if __name__ == "__main__":

    all_map = {}

    for dataset, K in (
                        ("load_iris",(2, 4)),
                        ("load_wine",(4, 6, 10, 13)),
                        ("load_breast_cancer",(5, 10, 15, 20, 32)),
                    ):

        x, y = getattr(datasets,dataset)(return_X_y=True, as_frame=False)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)
        print(dataset)

        for k in K:
            for transformer in (LDA,PCA):
                x_train_transformed, x_test_transformed = preprocess_data(transformer,x_train,x_test,y_train,k)
                for i in range(2):
                    key = [dataset]
                    title = transformer.__name__ + ", "
                    key.append(transformer.__name__)

                    if i == 0:
                        # zero mean, one variance
                        clf = make_pipeline(StandardScaler(), SVC())
                        key.append("SVC")
                        title += "SVC, "
                    elif i == 1:
                        clf = make_pipeline(RandomForestClassifier())
                        key.append("RFC")
                        title += "RFC, "

                    clf.fit(x_train_transformed, y_train)
                    y_pred = clf.predict(x_test_transformed)
                    
                    cm = Metric.confusion_matrix(y_test, y_pred)
                    title += f"k = {k} accuracy = {(cm[1]):.02f}"
                    print(f"\t\t{title}")

                    key.append(k)
                    all_map = add_key(all_map, tuple(key), cm[1:])

                    if k == 2:
                        plot_classification_region(clf, x_train_transformed, y_train, title=title, filename=key)
    
    df = pd.DataFrame.from_dict(all_map, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, name=["dataset", "transformer", "classifier", "k"])
    
    df = df[0].apply(pd.Series)
    df.columns = ["total_accuracy", "accuracy", "precision", "recall", "f1"]

    final_df = [df.total_accuracy,]

    for i in df.columns[1:]:
        final_df.append(df[i].apply(pd.Series))

    df = pd.concat(final_df, axis=1, keys=list(df.columns))
    df.columns.names = ["Metrics", "Classes"]
    df = df.apply(pd.to_numeric)

    df.style.format("{:.3f}",na_rep="").to_excel("test.xlsx")
