import numpy as np
import torch
import torch.nn as nn
from client import  local_train_classifier
from server import  Fed_avg_aggregate_classifier,Fed_avg_top_k_gradient
from models import LoRALinear,LinearClassifier
from argument_parser import get_parser
from utils import  make_dataset,split_iid,split_noniid_dirichlet, evaluate,set_seed,topk_sparse
from tqdm import tqdm
import logging
from config import SEED
import os

# Custom FileHandler that flushes after each log message
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Set up logging with the custom handler
def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = FlushFileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def main():

    set_seed(SEED)
    parser= get_parser()
    args = parser.parse_args()

    TOP_K=512 #768
    ### MAKE DATASET ####################################################################
    embeddings, labels = make_dataset(args.data_type)
    test_embs, test_labels = make_dataset(data_type='eval')
    #### SPLITTING THE DATASET USING IID or NON-IID DISTRIBUTION ########################
    num_classes =args.num_classes
    if args.distribution =='iid':
        client_emb,client_labels=split_iid(embeddings,labels,args.num_clients)
    elif args.distribution =='non_iid':
        client_emb,client_labels=split_noniid_dirichlet(embeddings,labels,args.num_clients,args.alpha)

    
    # if we change the client participation rate
    if args.is_part_rate:
        n_samples =int(args.part_rate*len(client_labels))
        indices = np.random.choice(len(client_emb), size=n_samples, replace=False)
        
        client_emb,client_labels =[client_emb[idx] for idx in indices],[client_labels[idx] for idx in indices]

    device= args.device
    device= args.device
    global_classifier = LinearClassifier(TOP_K, num_classes).to(device)
    global_state = global_classifier.state_dict()
    k_ratio =0.1 #10%
    test_accuracy_per_round=[]
    for r in tqdm(range(args.num_rounds)):

        local_states, client_sizes = [], []

        for emb, labls in zip(client_emb,client_labels):
            state, n = local_train_classifier(emb, labls, global_state, device, num_classes,method='top_k_sparse')

            sparse_delta = {n: topk_sparse(d, k_ratio) for n,d in state.items()}

            # here we append the sparsed deltas  to be send to the server.
            local_states.append(sparse_delta)
            client_sizes.append(n)
        

        global_state = Fed_avg_top_k_gradient(global_state,local_states, client_sizes) 
        global_classifier.load_state_dict(global_state)
        # here we dont do any random projection (P=None) nor eigenvector calculation (U=None)
        U=None;P=None
        acc = evaluate(global_classifier, U, test_embs, test_labels, device,args.method,P)

        test_accuracy_per_round.append(acc)
        logging.info(f"Round {r+1}/{args.num_rounds} complete. - Test Accuracy: {acc:.4f}")
    logging.info(f"Experiment for classes={args.num_classes}, with clients={args.num_clients} clients and test_accuracy={test_accuracy_per_round}")

if __name__ == "__main__":
    k_ratio =0.1 # 10%
    parser= get_parser()
    args = parser.parse_args()
    log_path=f"logs/{args.num_classes}_{args.method}_grad_sparse_10_percent_{args.log_file}"
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Deleted existing log file: {log_path}")
    setup_logging(log_path)
    main()