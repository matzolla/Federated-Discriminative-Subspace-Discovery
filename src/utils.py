import numpy as np
from collections import defaultdict
import joblib, glob 
from tqdm import tqdm
import torch
import random
import os
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


######### MAKING THE DATASETS #############################################
def make_dataset(data_type):
    if data_type=='train':
        files = glob.glob(r"data/Mvit_embds/TOYOTA/toyota_train/*.joblib") #/Users/tonystark/Desktop/fed_angular_pca/data/Resnet18_embds/UCF101/train_embeddings
    else:
        files = glob.glob(r"data/Mvit_embds/TOYOTA/toyota_test/*.joblib") 
    embs, labels = [], [] 
    for f in tqdm(files): 
        obj = joblib.load(f) 
        embs.append(obj["embedding"]) 
        labels.append(obj["label"]) 
    embs = np.vstack(embs).astype(np.float32) 
    labels = np.array(labels)
    return embs,labels


############### PERFORMING SINGULAR VALUE DECOMPOSITION OF THE SCATTER-MATRIX#######
def topk_eig(mat,k):
    """
    Return top-k eigenvectors (columns) and eigenvalues of symmetric matrix mat.
    """
    vals,vecs = np.linalg.eigh(mat)
    idx = np.argsort(vals)[::-1][:k]

    return vecs[:, idx], vals[idx]


############### RANDOM PROJECTION OF THE SCATTER MATRIX #########################
def approximate_scatter(Z, method='projection', dim=256, random_state=None):
    """
    Compute a low-dimensional approximation of covariance from embeddings Z.

    Parameters:
    -----------
    Z : np.ndarray
        Original scatter matrix, shape (d_orig, d_orig)
    method : str
        'projection' -> project to lower dim before covariance
        'topk_eigen' -> compute top-k eigenvectors of full covariance
    dim : int
        Target dimension (projection dim or top-k eigenvectors)
    random_state : int or None
        Seed for reproducibility (only for projection)

    Returns:
    --------
    If method='projection':
        cov_approx : np.ndarray, shape (dim, dim)
    If method='topk_eigen':
        eigvecs_topk : np.ndarray, shape (d_orig, dim)
        eigvals_topk : np.ndarray, shape (dim,)
    """
    np.random.seed(random_state)
    n_samples, d_orig = Z.shape
    Z_centered = Z - Z.mean(axis=0, keepdims=True)

    if method == 'projection':
        # Random Gaussian projection
        P = np.random.randn(d_orig, dim) / np.sqrt(dim)
        Z_proj = Z_centered @ P  # shape (n_samples, dim)
        cov_approx = Z_proj.T @ Z_proj  # shape (dim, dim)
        return cov_approx,P

    elif method == 'topk_eigen':
        # Full covariance
        C = Z_centered.T @ Z_centered  # shape (d_orig, d_orig)
        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(C)  # ascending order
        eigvals = eigvals[::-1]  # descending
        eigvecs = eigvecs[:, ::-1]
        eigvals_topk = eigvals[:dim]
        eigvecs_topk = eigvecs[:, :dim]  # shape (d_orig, dim)
        return eigvecs_topk, eigvals_topk

    else:
        raise ValueError("method must be 'projection' or 'topk_eigen'")
    
####################### THIS IS TO NORMALIZE THE EMBEDDINGS #########################
def normalize_rows(x,eps=1e-9):

    norms=np.linalg.norm(x,axis=1,keepdims=True)

    return x/(norms+eps)

################### IID DATASET SPLITTING ###########################################
def split_iid(embs, labels, num_clients=10, seed=0):
    np.random.seed(seed)
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    embs, labels = embs[idx], labels[idx]

    client_embs, client_labels = [], []
    sizes = np.full(num_clients, len(labels)//num_clients)
    sizes[:len(labels) % num_clients] += 1  # handle remainder
    start = 0
    for s in sizes:
        client_embs.append(embs[start:start+s])
        client_labels.append(labels[start:start+s])
        start += s
    return client_embs, client_labels

#################### NON-IID DATASET SPLITTING #########################################
def split_noniid_dirichlet(embs, labels, num_clients=10, alpha=0.2, seed=0):
    np.random.seed(seed)
    num_classes = labels.max() + 1
    client_indices = [[] for _ in range(num_clients)]

    # For each class, split its samples across clients using Dirichlet proportions
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(idx_c)).astype(int)

        # Fix rounding issues so total = len(idx_c)
        while proportions.sum() < len(idx_c):
            proportions[np.random.randint(num_clients)] += 1
        while proportions.sum() > len(idx_c):
            proportions[np.random.randint(num_clients)] -= 1

        start = 0
        for i, p in enumerate(proportions):
            client_indices[i].extend(idx_c[start:start+p])
            start += p

    # Build final splits
    client_embs, client_labels = [], []
    for idxs in client_indices:
        client_embs.append(embs[idxs])
        client_labels.append(labels[idxs])
    return client_embs, client_labels

###################### MODEL EVALUATION #####################################################
def evaluate(global_classifier, U, test_embs, test_labels, device,method,P):
    global_classifier.eval()
    with torch.no_grad():
        if (method == 'angular_pca'):
            Z=test_embs@P
            Z = Z @ U # project test embeddings
        else:
            Z= test_embs                  
        X = torch.tensor(Z, dtype=torch.float32).to(device)
        y = torch.tensor(test_labels, dtype=torch.long).to(device)

        logits = global_classifier(X)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean().item()
    return acc


#### evaluation for ablation ########

def ablation_evaluate(global_classifier,test_embs, test_labels, device,method,P):
    global_classifier.eval()
    with torch.no_grad():
        if (method == 'angular_pca'):
            Z=test_embs@P
        else:
            Z= test_embs                  
        X = torch.tensor(Z, dtype=torch.float32).to(device)
        y = torch.tensor(test_labels, dtype=torch.long).to(device)

        logits = global_classifier(X)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean().item()
    return acc

####################### SETTING THE SEEDS FOR OUR EXPERIMENTS ################################
def set_seed(seed: int = 0):
    # Python built-in RNG
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior (may slow down a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # (Optional) For dataloader workers
    os.environ["PYTHONHASHSEED"] = str(seed)

####################### QUANTIZING THE SCATTER MATRIX #########################################
def quantize_scatter(C, method='float16', upper_triangle=False):
    """
    Computes covariance matrix Z^T Z and quantizes it.
    
    C: tensor of shape (d, d)
    method: 'float16', '8bit', '4bit'
    upper_triangle: whether to return only upper triangle
    """
    
    if upper_triangle:
        idx = torch.triu_indices(C.size(0), C.size(1))
        C = C[idx[0], idx[1]]  # flatten upper triangle
    
    if method == 'float16':
        C = C.astype(np.float16)
    elif method == '8bit':
        min_val, max_val = C.min(), C.max()
        C = np.round(((C - min_val) / (max_val - min_val) * 255)).astype(np.uint8)
        # optionally return min_val/max_val for reconstruction
        return C
    elif method == '4bit':
        min_val, max_val = C.min(), C.max()
        C = np.round(((C - min_val) / (max_val - min_val) * 15)).astype(np.uint8)
        # 2 values per byte, bit-packing could be applied
        return C
    return C

######## QUANTIZING THE LINEAR CLASSIFIER (MODEL COMPRESSION) ##############################


class ModelQuantizer:
    """
    Flexible model quantizer with proper 4-bit, 8-bit, and 16-bit support.

    Usage:
        q = ModelQuantizer(num_bits=4)
        q_state = q.quantize(model.state_dict())
        dq_state = q.dequantize(q_state)
    """
    def __init__(self, num_bits=8):
        assert num_bits in [4, 8, 16, 32], "num_bits must be one of [4, 8, 16, 32]"
        self.num_bits = num_bits

    def quantize(self, state_dict):
        q_state = {}

        # --------- 16-bit float ---------
        if self.num_bits == 16:
            for k, v in state_dict.items():
                q_state[k] = v.half() if torch.is_floating_point(v) else v
            q_state["_dtype"] = "float16"
            return q_state

        # --------- 4-bit or 8-bit integer ---------
        for k, v in state_dict.items():
            if not torch.is_floating_point(v):
                q_state[k] = v
                continue

            # ---- special 4-bit symmetric per-row for Linear weights ----
            if self.num_bits == 4 and v.ndim == 2 and v.shape[1] > 16:
                # treat as [out, in] weight matrix
                max_abs = v.abs().max(dim=1).values      # [out]
                scale = max_abs / 7                      # 4-bit symmetric
                scale = scale[:, None]                   # broadcast
                q = torch.round(v / scale).clamp(-8, 7)
                q_state[k] = q.to(torch.int8)
                q_state[k + "_scale"] = scale.squeeze(1)
                q_state[k + "_zp"] = 0.0                 # symmetric
                continue

            # ---- generic per-tensor affine quantization (8-bit or fallback) ----
            v_min, v_max = v.min(), v.max()
            if v_min == v_max:
                scale = torch.tensor(1.0)
                zp = torch.tensor(0.0)
                q = torch.zeros_like(v, dtype=torch.int8)
            else:
                scale = (v_max - v_min) / (2**self.num_bits - 1)
                zp = (-v_min / scale).round()
                q = ((v / scale) + zp).round().clamp(0, 2**self.num_bits - 1)
            q_state[k] = q.to(torch.int8)
            q_state[k + "_scale"] = scale
            q_state[k + "_zp"] = zp

        q_state["_dtype"] = f"int{self.num_bits}"
        return q_state

    def dequantize(self, q_state):
        dq_state = {}

        # For 16-bit, just convert back to float32
        if q_state.get("_dtype", "") == "float16":
            for k, v in q_state.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                    dq_state[k] = v.float()
                elif not k.startswith("_"):
                    dq_state[k] = v
            return dq_state

        for k, v in q_state.items():
            # Skip non-tensor entries
            if not isinstance(v, torch.Tensor):
                continue

            # Skip scale/zero-point entries
            if k.endswith("_scale") or k.endswith("_zp") or k.startswith("_"):
                continue

            scale = q_state[k + "_scale"]
            zp = q_state.get(k + "_zp", 0.0)

            # if scale is 1D and matches first dim of v, expand
            if scale.ndim == 1 and scale.shape[0] == v.shape[0]:
                scale = scale[:, None]
            dq_state[k] = scale * (v.float() - zp)

        return dq_state


############## Top-k gradient sparsification ###############################

def topk_sparse(delta, k_ratio):
    flat = delta.flatten()
    k = max(1, int(k_ratio * flat.numel()))
    idx = flat.abs().topk(k).indices
    sparse = torch.zeros_like(flat)
    sparse[idx] = flat[idx]
    return sparse.view_as(delta)



####### THIS FUNCTIONS ARE NECESSARY FOR ANALYSIS AND EVIDENCE###############




# -----------------------------
# Compute SNR of a subspace
# -----------------------------
def compute_SNR(U, X, y):
    """
    U : (d, k) projection matrix
    X : (n, d) embeddings
    y : (n,) labels
    """
    mu = X.mean(axis=0)
    classes = np.unique(y)
    N = len(y)
    
    # Between-class scatter
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        muc = Xc.mean(axis=0)
        d = (muc - mu).reshape(-1,1)
        Sb += nc * (d @ d.T)

    # Total scatter
    St = np.cov(X, rowvar=False)

    # Within-class scatter
    Sw = St - Sb / N

    # Projected SNR
    num = np.trace(U.T @ Sb @ U)
    den = np.trace(U.T @ Sw @ U) + 1e-12
    return num / den


# --------------------------------
# Principal angle between subspaces
# --------------------------------
def mean_principal_angle(U1, U2):
    """
    Returns average principal angle between two subspaces
    """
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    angles = np.arccos(np.clip(s, -1, 1))
    return angles.mean()


# --------------------------------
#  Accuracy of linear probe
# --------------------------------
def linear_probe_accuracy(X_train,X_test, y,y_test, U=None):
    """
    Train linear classifier on projected data
    """
    if U is not None:
        X_proj = X_train @ U
        X_proj_test= X_test @ U
    else:
        X_proj = X_train
        X_proj_test= X_test
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_proj, y)
    preds = clf.predict(X_proj_test)
    return accuracy_score(y_test, preds)


# --------------------------------
#  Sweep over k and plot
# --------------------------------
def sweep_accuracy_vs_k(X,X_test, y,y_test, ks):
    d = X.shape[1]
    vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))
    idx = np.argsort(vals)[::-1]  # sort descending
    vecs = vecs[:, idx]

    accs = []
    snrs = []
    for k in ks:
        U = vecs[:, :k]
        acc = linear_probe_accuracy(X,X_test, y,y_test, U)
        snr = compute_SNR(U, X, y)
        accs.append(acc)
        snrs.append(snr)

    return accs, snrs




