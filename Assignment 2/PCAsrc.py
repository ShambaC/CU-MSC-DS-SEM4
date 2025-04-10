import numpy as np

def PCA(n_components: int, data: np.ndarray, standardize: bool) -> np.ndarray :

    # Data is assumed to contiain information row wise
    
    # Feature wise standardize the data
    if standardize :
        data = data.T
        for i in range(data.shape[0]) :
            data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])

        data = data.T

    # Covariance matrix computation
    covar_mat = np.zeros((data.shape[1], data.shape[1]))

    for i in range(data.shape[0]) :
        for j in range(data.shape[1]) :
            if j < i :
                continue

            col1 = data[:, i:i+1]
            col2 = data[:, j:j+1]

            cov = 0
            if standardize :
                cov = np.sum(col1 * col2) / data.shape[0] # Because standard distribution mean = 0
            else :
                cov = np.sum((col1 - np.mean(col1)) * (col2 - np.mean(col2))) / data.shape[0]
            covar_mat[i][j] = cov
            covar_mat[j][i] = cov

    eigenvalues, eigenvectors = np.linalg.eigh(covar_mat)
    eigen_val_order = np.argsort(eigenvalues)[::-1]

    eigenvec_selected = eigenvectors[:, [eigen_val_order[i] for i in range(n_components)]]

    reduced_data = np.matmul(data, eigenvec_selected)

    return reduced_data

if __name__ == "__main__" :
    import pandas as pd

    df = pd.read_csv("IRIS/iris.data", header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type'])
    df.drop(["type"], axis=1, inplace=True)

    print(df.head(10))

    n_components = 2
    standardize = True

    reduced_iris = PCA(n_components, df.to_numpy(), standardize)
    print(reduced_iris[:10])