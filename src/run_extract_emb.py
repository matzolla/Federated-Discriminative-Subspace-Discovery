import numpy as np
import torch
import torch.nn.functional as F
from data import get_dataloader
from models import video_Mvit,resnet3D
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import joblib
import os
from argument_parser import get_parser


def main():

    parser= get_parser()
    args = parser.parse_args()



    device=args.device
    output_dir = r"C:\Users\23113181\Desktop\Federated_exp\fed_angular_pca\data\Resnet18_embds\TOYOTA\test_embeddings"
    path= args.path
    # Load data
    train_loader,_ = get_dataloader(data_path=path,batch_size=args.batch_size,clip_length=args.clip_size,num_clips=1,num_workers=0)

    model= resnet3D(101).to(device)

    model.eval()

    saved=0

    with torch.no_grad():
        for idx, (clip,labels) in enumerate(tqdm(train_loader)):
            clip = clip.to(device).float()
            encoder= create_feature_extractor(model,return_nodes={'r3d_18.flatten':'r3d_18.flatten'}) ## replace fc with 'head.0', if working with Mvit
            ## features extracted from the encoder
            emb=encoder(clip)

            emb = emb['r3d_18.flatten'].cpu().numpy()
            labels= labels.numpy()

            for j in range(len(labels)):
                fname=f"vid_{saved:06d}.joblib"
                joblib.dump({"embedding": emb[j], "label": int(labels[j])}, os.path.join(output_dir, fname))
                saved += 1



if __name__ == "__main__":
    main()