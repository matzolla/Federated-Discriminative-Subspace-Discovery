import numpy as np
import torch
from client import compute_scatter, local_train_classifier
from server import aggregate_subspace, Fed_avg_aggregate_classifier
from models import LinearClassifier
from argument_parser import get_parser
from utils import  make_dataset,split_iid,split_noniid_dirichlet, evaluate,set_seed,approximate_scatter,quantize_scatter
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
    if args.ablation =="k":
        ## vary the k (top-k) values (n_alpha)
        element_to_vary=np.arange(0.2,1,0.1)
    #### INITIALIZE THE GLOBAL CLASSIFIER ###############################################
    elif args.ablation =="m":
        ## vary the random projection dimension values 
        element_to_vary=np.arange(200,768,100)
    
    num_classes =args.num_classes
    # working with our angular pca approach
    device= args.device
    test_accuracy_per_element=[]
    for element in element_to_vary:
        
        if args.ablation=="k":
            # we're varying the eigen vectors top-k
            n_alpha=element
            TOP_K=768
        else:
            # we're varying the random projection dimension
            TOP_K=element
            n_alpha=100/TOP_K# awe fix k to 100 for varying values of m
        
        global_classifier = LinearClassifier(int(n_alpha*TOP_K), num_classes).to(device)
        global_state = global_classifier.state_dict()

        #### FEDERATED LEARNING #############################################################
        
        for r in tqdm(range(args.num_rounds)):
            scatter_mats, counts = [], []
            local_states, client_sizes = [], []
            ### Here we start by submitting the scatter matrix of all the clients for aggregation
            for emb, labls in zip(client_emb,client_labels):

                if args.method== "angular_pca":
                    Sm = compute_scatter(emb) ## here the scatter matrix is of dimension (768,768)
                    
                    Sm,P= approximate_scatter(Sm,method='projection',dim=int(TOP_K),random_state=SEED) ## we project the scatter matrix to a lower dimension
                    Sm=quantize_scatter(Sm,method=args.quantize) ## Then we quantize the reduced scatter matrix
                    scatter_mats.append(Sm); counts.append(len(labls))
            
            if args.method == "angular_pca":
                U = aggregate_subspace(scatter_mats, counts, int(n_alpha*TOP_K))

            ### Then we federate the weghts of our classifier
            for emb, lab in zip(client_emb, client_labels):
                
                if args.method=="angular_pca":
                    Z=emb@P # reduce the dimension (n,Top_k) to a lower dimension using random projection
                    Z = Z @ U
                else:
                    # no projection (Here U and P are not used in any case)
                    Z= emb
                    U= None
                    P= None
                
                state, n = local_train_classifier(Z, lab, global_state, device, num_classes)
                local_states.append(state); client_sizes.append(n)
            
            global_state = Fed_avg_aggregate_classifier(local_states, client_sizes,bits=None) #bits is set to None because we're not quantizing the model here [maybe later]
            global_classifier.load_state_dict(global_state)
            
            acc = evaluate(global_classifier, U, test_embs, test_labels, device,args.method,P)
        test_accuracy_per_element.append(acc)
            #logging.info(f"Round {r+1}/{args.num_rounds} complete. - Test Accuracy: {acc:.4f}")
    logging.info(f"Experiment for classes={args.num_classes}, with clients={args.num_clients} clients and test_accuracy={test_accuracy_per_element} with element_to_vary={element_to_vary}")    
    
if __name__ == "__main__":
    parser= get_parser()
    args = parser.parse_args()
    log_path=f"logs/ablation_varying_{args.ablation}_{args.log_file}"
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Deleted existing log file: {log_path}")
    setup_logging(log_path)
    main()