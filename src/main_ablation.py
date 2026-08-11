import numpy as np
import torch
from client import compute_scatter, local_train_classifier
from server import aggregate_subspace, Fed_avg_aggregate_classifier
from models import LinearClassifier
from argument_parser import get_parser
from utils import  make_dataset,split_iid,split_noniid_dirichlet, evaluate,set_seed,approximate_scatter,ablation_evaluate
from tqdm import tqdm
from config import SEED
import logging
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
    
    ### MAKE DATASET ####################################################################
    embeddings, labels = make_dataset(args.data_type)
    test_embs, test_labels = make_dataset(data_type='eval')
    #### SPLITTING THE DATASET USING IID or NON-IID DISTRIBUTION ########################

    if args.distribution =='iid':
        client_emb,client_labels=split_iid(embeddings,labels,args.num_clients)
    elif args.distribution =='non_iid':
        client_emb,client_labels=split_noniid_dirichlet(embeddings,labels,args.num_clients,args.alpha)

    
    #### INITIALIZE THE GLOBAL CLASSIFIER ###############################################

    num_classes =args.num_classes
 
    TOP_K = args.top_k


    device= args.device
    global_classifier = LinearClassifier(int(TOP_K), num_classes).to(device)
    global_state = global_classifier.state_dict()

    #### FEDERATED LEARNING #############################################################
    test_accuracy_per_round=[]
    for r in tqdm(range(args.num_rounds)):
        scatter_mats, counts = [], []
        local_states, client_sizes = [], []
        ### Here we start by submitting the scatter matrix of all the clients for aggregation
        for emb, labls in zip(client_emb,client_labels):

            if args.method== "angular_pca":
                Sm = compute_scatter(emb) ## here the scatter matrix is of dimension (768,768)
                
                Sm,P= approximate_scatter(Sm,method='projection',dim=int(TOP_K),random_state=SEED) ## we project the scatter matrix to a lower dimension
                counts.append(len(labls))
        # we don't use scatter matrices here the objective is to see if we actually need to send a covariance matrix or not
        # we perform a random projection and train the local linear classifier on the lower dimensional space.

        ### Then we federate the weghts of our classifier
        for emb, lab in zip(client_emb, client_labels):
            
            if args.method=="angular_pca":
                Z=emb@P # reduce the dimension (n,Top_k) to a lower dimension using random projection
            else:
                # no projection (Here U and P are not used in any case)
                Z= emb
                U= None
                P= None
            
            state, n = local_train_classifier(Z, lab, global_state, device, num_classes)
            local_states.append(state); client_sizes.append(n)
        
        global_state = Fed_avg_aggregate_classifier(local_states, client_sizes,bits=None) #bits is set to None because we're not quantizing the model here [maybe later]
        global_classifier.load_state_dict(global_state)
        
        acc = ablation_evaluate(global_classifier, test_embs, test_labels, device,args.method,P)
        test_accuracy_per_round.append(acc)
        logging.info(f"Round {r+1}/{args.num_rounds} complete. - Test Accuracy: {acc:.4f}")
    logging.info(f"Experiment for classes={args.num_classes}, with clients={args.num_clients} clients and test_accuracy={test_accuracy_per_round}")    
    
if __name__ == "__main__":
    parser= get_parser()
    args = parser.parse_args()
    log_path=f"logs/Ablation_{args.num_classes}_{args.method}_{args.log_file}"
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Deleted existing log file: {log_path}")
    setup_logging(log_path)
    main()