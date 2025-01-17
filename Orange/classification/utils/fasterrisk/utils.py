import numpy as np
from itertools import product
import requests

def get_groupIndex_to_featureIndices(featureIndex_to_groupIndex):
    groupIndex_to_featureIndices = {}
    for featureIndex, groupIndex in enumerate(featureIndex_to_groupIndex):
        if groupIndex not in groupIndex_to_featureIndices:
            groupIndex_to_featureIndices[groupIndex] = set()
        groupIndex_to_featureIndices[groupIndex].add(featureIndex)
    return groupIndex_to_featureIndices

def get_support_indices(betas):
    return np.where(np.abs(betas) > 1e-9)[0]

def get_nonsupport_indices(betas):
    return np.where(np.abs(betas) <= 1e-9)[0]

def normalize_X(X):
    X_mean = np.mean(X, axis=0)
    X_norm = np.linalg.norm(X-X_mean, axis=0)
    scaled_feature_indices = np.where(X_norm >= 1e-9)[0]
    X_normalized = X-X_mean
    X_normalized[:, scaled_feature_indices] = X_normalized[:, scaled_feature_indices]/X_norm[[scaled_feature_indices]]
    return X_normalized, X_mean, X_norm, scaled_feature_indices

def compute_logisticLoss_from_yXB(yXB):
    # shape of yXB is (n, )
    return np.sum(np.log(1.+np.exp(-yXB)))

def compute_logisticLoss_from_ExpyXB(ExpyXB):
    # shape of ExpyXB is (n, )
    return np.sum(np.log(1.+np.reciprocal(ExpyXB)))

def compute_logisticLoss_from_betas_and_yX(betas, yX):
    # shape of betas is (p, )
    # shape of yX is (n, p)
    yXB = yX.dot(betas)
    return compute_logisticLoss_from_yXB(yXB)

def compute_logisticLoss_from_X_y_beta0_betas(X, y, beta0, betas):
    XB = X.dot(betas) + beta0
    yXB = y * XB
    return compute_logisticLoss_from_yXB(yXB)

def convert_y_to_neg_and_pos_1(y):
    y_max, y_min = np.min(y), np.max(y)
    y_transformed = -1 + 2 * (y-y_min)/(y_max-y_min) # convert y to -1 and 1
    return y_transformed

def isEqual_upTo_8decimal(a, b):
    if np.isscalar(a):
        return abs(a - b) < 1e-8
    return np.max(np.abs(a - b)) < 1e-8

def isEqual_upTo_16decimal(a, b):
    if np.isscalar(a):
        return abs(a - b) < 1e-16
    return np.max(np.abs(a - b)) < 1e-16

def insertIntercept_asFirstColOf_X(X):
    n = len(X)
    intercept = np.ones((n, 1))
    X_with_intercept = np.hstack((intercept, X))
    return X_with_intercept

def get_all_product_booleans(sparsity=5):
    # build list of lists:
    all_lists = []
    for i in range(sparsity):
        all_lists.append([0, 1])
    all_products = list(product(*all_lists))
    all_products = [list(elem) for elem in all_products]
    return np.array(all_products)

def download_file_from_google_drive(id, destination):
    # link: https://stackoverflow.com/a/39225272/5040208
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id , 'confirm': 1 }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    # link: https://stackoverflow.com/a/39225272/5040208
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    # link: https://stackoverflow.com/a/39225272/5040208
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def check_bounds(bound,  bound_name, num_features):
    if isinstance(bound, (float, int)):
        assert bound >= 0 if bound_name == "ub" else bound <= 0, f"{bound_name} needs to be >= 0" if bound_name == "ub" else f"{bound_name} needs to be <= 0"
    elif isinstance(bound, list):
        bound = np.asarray(bound)
        assert len(bound) == num_features, f"{bound_name}s for the features need to have the same length as the number of features"
        assert np.all(bound >= 0 if bound_name == "ub" else bound <= 0), f"all of {bound_name}s needs to be >= 0" if bound_name == "ub" else f"all of {bound_name}s needs to be <= 0"
    else:
        raise ValueError(f"{bound_name} needs to be a float, int, or list")
    
    return bound