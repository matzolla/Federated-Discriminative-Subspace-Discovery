import numpy as np
import torch
from client import compute_scatter, local_train_classifier
from server import aggregate_subspace, Fed_avg_aggregate_classifier
from models import LinearClassifier,OneHiddenLayerClassifier,TwoHiddenLayerClassifier
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

    # if we change the client participation rate
    if args.is_part_rate:
        n_samples =int(args.part_rate*len(client_labels))
        indices = np.random.choice(len(client_emb), size=n_samples, replace=False)
        
        client_emb,client_labels =[client_emb[idx] for idx in indices],[client_labels[idx] for idx in indices]
    #### INITIALIZE THE GLOBAL CLASSIFIER ###############################################

    num_classes =args.num_classes
    if args.method=="angular_pca":
        # working with our angular pca approach
        TOP_K = args.top_k
        n_alpha=0.50
    else:
        # we consider just the number of embeddings that we have
        TOP_K=768 # for MVit dimension =768
        n_alpha=1
    device= args.device
    if args.classifier=="linear":
        global_classifier = LinearClassifier(int(n_alpha*TOP_K), num_classes).to(device)
    elif args.classifier=="onelayer":
        global_classifier = OneHiddenLayerClassifier(int(n_alpha*TOP_K),args.hidden_node,num_classes).to(device)
    elif args.classifier=="twolayer":
        global_classifier=TwoHiddenLayerClassifier(int(n_alpha*TOP_K),args.hidden_node,args.hidden_node,num_classes).to(device)
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
                Sm=quantize_scatter(Sm,method=args.quantize) ## Then we quantize the reduced scatter matrix
                scatter_mats.append(Sm); counts.append(len(labls))
        
        if args.method == "angular_pca":
            ## compute the subspace after r rounds
            if r%args.period==0:
                print(f"computing the subspace at round {r}")
                U_subspace = aggregate_subspace(scatter_mats, counts, int(n_alpha*TOP_K))

            
            U=U_subspace

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
            
            state, n = local_train_classifier(Z, 
                                              lab, 
                                              global_state, 
                                              device, 
                                              num_classes,
                                              class_type=args.classifier,
                                              in_dim=int(n_alpha*TOP_K),
                                              hidden_dim=args.hidden_node)
            
            local_states.append(state); client_sizes.append(n)
        
        global_state = Fed_avg_aggregate_classifier(local_states, client_sizes,bits=None) #bits is set to None because we're not quantizing the model here [maybe later]
        global_classifier.load_state_dict(global_state)
        
        acc = evaluate(global_classifier, U, test_embs, test_labels, device,args.method,P)
        test_accuracy_per_round.append(acc)
        logging.info(f"Round {r+1}/{args.num_rounds} complete. - Test Accuracy: {acc:.4f}")
    logging.info(f"Experiment for classes={args.num_classes}, with clients={args.num_clients} clients and test_accuracy={test_accuracy_per_round}")    
    
if __name__ == "__main__":
    parser= get_parser()
    args = parser.parse_args()
    log_path=f"logs/{args.num_classes}_{args.method}_{args.log_file}"
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Deleted existing log file: {log_path}")
    setup_logging(log_path)
    main()