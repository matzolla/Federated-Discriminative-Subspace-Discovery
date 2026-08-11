import numpy as np
import torch
from client import compute_scatter, local_train_classifier
from server import aggregate_subspace, Fed_avg_aggregate_classifier
from models import LinearClassifier
from argument_parser import get_parser
from utils import  make_dataset,set_seed,sweep_accuracy_vs_k
from tqdm import tqdm
from config import SEED
import joblib,glob
import matplotlib.pyplot as plt

def main():
    set_seed(SEED)
    parser= get_parser()
    args = parser.parse_args()
    paths=[
        [
        ("data/Mvit_embds/TOYOTA/toyota_train/*.joblib","data/Mvit_embds/TOYOTA/toyota_test/*.joblib","TOYOTA"),
        ("data/Mvit_embds/UCF101/ucf101_train/*.joblib","data/Mvit_embds/UCF101/ucf101_test/*.joblib","UCF101"),
        ("data/Mvit_embds/HMDB51/hmdb51_train/*.joblib","data/Mvit_embds/HMDB51/hmdb51_test/*.joblib","HMDB51")
        ],
        [
            ("data/Resnet18_embds/TOYOTA/train_embeddings/*.joblib","data/Resnet18_embds/TOYOTA/test_embeddings/*.joblib","TOYOTA"),
            ("data/Resnet18_embds/UCF101/train_embeddings/*.joblib","data/Resnet18_embds/UCF101/test_embeddings/*.joblib","UCF101"),
            ("data/Resnet18_embds/HMDB51/train_embeddings/*.joblib","data/Resnet18_embds/HMDB51/test_embeddings/*.joblib","HMDB51")

        ]
        ]
    model_name=['MviT','ResNet3D-18']
    def make_dataset(path):

        files = glob.glob(path) #downsample_ucf101_16x12
        embs, labels = [], [] 
        for f in tqdm(files): 
            obj = joblib.load(f) 
            embs.append(obj["embedding"]) 
            labels.append(obj["label"]) 
        embs = np.vstack(embs).astype(np.float32) 
        labels = np.array(labels)
        
        return embs,labels
            # Plot
    
    
    #plt.figure(figsize=(20,9))
    fig, axes =plt.subplots(1,len(paths),figsize=(20,9))
    for idx in range(len(paths)):
        ks = [[8,16,32, 64, 128, 200,256, 310,512, 768],[8,16,32, 64, 128, 200,256, 310,512]]
        for path_train,path_test,data_name in paths[idx]:
        ### MAKE DATASET ####################################################################
            embs, labels = make_dataset(path_train)
            embs_test, labels_test = make_dataset(path_test)
            
            accs, _ = sweep_accuracy_vs_k(embs,embs_test, labels,labels_test, ks[idx])

            axes[idx].plot(ks[idx], accs, "--",linewidth=5, label=f"{data_name}")
            
            # Find the index of k = 256
            if 256 in ks[idx]:
                idy = ks[idx].index(256)
                axes[idx].plot(ks[idx][idy], accs[idy], 'r*', markersize=30)
        axes[idx].grid(True,color='black')
        axes[idx].tick_params(axis='both',which='major',labelsize=35)
        axes[idx].set_title(model_name[idx],fontsize=35)

    
    axes[0].set_ylabel("Accuracy",fontsize=35)
    axes[0].legend(fontsize=26, loc="lower right")
    fig.supxlabel("Top-$k$ dimensions",fontsize=35,y=-0.01)
    #plt.ylabel("Accuracy",fontsize=35)
    
   
    #plt.legend(fontsize=26)
    # Save the figure
    fig.savefig("evidence_plot.png")
    #plt.show()
    #### SPLITTING THE DATASET USING IID or NON-IID DISTRIBUTION ########################


if __name__ == "__main__":
    main()