import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import normalize_rows,topk_sparse
from models import LinearClassifier,LoRALinear,OneHiddenLayerClassifier,TwoHiddenLayerClassifier
from config import BATCH_SIZE, LR, LOCAL_EPOCHS

def compute_scatter(embeddings):
    """Compute angular scatter for local embeddings."""
    Z = normalize_rows(embeddings)
    return Z.T @ Z

def local_train_classifier(X, y, init_state_dict, device, 
                           num_classes, 
                           method=None,
                           class_type=None,
                           in_dim=None,
                           hidden_dim=None):
    if method=="lora":
        # Converting the linear classifier into a low-rank approximation
        base= nn.Linear(X.shape[1], num_classes).to(device)
        model=LoRALinear(base,r=32, alpha=64).to(device)
    else:
        if class_type =="linear":
            model = LinearClassifier(X.shape[1], num_classes).to(device)
        elif class_type=="onelayer":
            model= OneHiddenLayerClassifier(in_dim,hidden_dim,num_classes)
        elif class_type=="twolayer":
            model=TwoHiddenLayerClassifier(in_dim,hidden_dim,hidden_dim,num_classes)
    ##### #######################
    model.load_state_dict(init_state_dict)
    opt = optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    dataset = list(zip(X, y))
    for ep in range(LOCAL_EPOCHS):
        np.random.shuffle(dataset)
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i:i+BATCH_SIZE]
            xb = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32).to(device)
            yb = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.long).to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    #### if the method is top_k        
    if method =='top_k_sparse':
            # compute dense delta
        delta = {}
        for (n, p), (_, g) in zip(model.state_dict().items(), init_state_dict.items()):
            delta[n] = p.data - g.data
        return delta ,len(y)
    return model.state_dict(), len(y)

