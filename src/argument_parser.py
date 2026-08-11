import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description="My Program Description"
    )
    
    ###### SYSTEM SETTING ######
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="setting the device on which we're training our experiments [cuda,cpu]"
    )

    parser.add_argument('--log_file', 
                        type=str, 
                        default='results.log', 
                        help='log file')

    ###### THE INPUTS ##########
    parser.add_argument(
        "--data_type",
        type=str,
        default="train",
        help="If we're either in training or evaluation mode"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=r"C:\Users\23113181\Downloads\results_and_graphics_for_first_paper\toyota_data\test",
        help="Path to the input file"
    )

    parser.add_argument(
      "--batch_size",
      type=int,
      default= 8,
      help="The batch size used for our experiments and also to extract the embeddings"
    )
    parser.add_argument(
        "--clip_size",
        type=int,
        default=16,
        help="The lenghth of clip used in our experiments (16 frames per clips)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="result.txt",
        help="Path to the output file (default: result.txt)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode"
    )

    parser.add_argument(
        "--distribution",
        type=str,
        default="iid",
        help="distribution could be iid or non_iid"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="the alpha value necessary to create non-iid distribution (using dirichlet approach)"
    )

    ########### CLIENTS ############################

    parser.add_argument(
        "--num_clients",
        type =int,
        default=20,
        help="The number of clients to federated"
    )

    parser.add_argument(

        "--quantize",
        type=str,
        default="4bit",
        help="This help us quantize the scatter matrix into a lower bits, therefore considerably reducing the communication cost"
    )

    parser.add_argument(
        "--model_quantized_bits",
        type=int,
        default=8,
        help="This is to quantize the model base on the number of bits used [4,8,16]"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="angular_pca",
        help="either 'angular_pca' or 'vanilla' methods" 
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=31,
        help="The number of classes used by our dataset UCF101 (101) , HMDB51 (51)"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="The number of eigen vectors that we are selecting (Top-k) k={64,128,256 etc....}"
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Number of federated rounds"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="m",
        help="either varying m or k"
    )

    parser.add_argument(
        "--part_rate",
        type=float,
        default=0.2,
        help="The participation rate of clients in federated learning [0.1,1] 1 is full participation"
    )

    parser.add_argument(
        "--is_part_rate",
        type=bool,
        default=False,
        help="set to True if we want clients to participate with a certain probability "
    )
    parser.add_argument(
        "--period",
        type=int,
        default=1,
        help="The frequency at which we recompute the subspace U "
    )


    ###### The Classifier ###############
    parser.add_argument(
        "--classifier",
        type=str,
        default="twolayer",
        help="The choice of the classifier it could be linear, onelayer,twolayer"
    )
    parser.add_argument(
        "--hidden_node",
        type=int,
        default=30,
        help="the number of hidden nodes of the hidden layers"
    )
    return parser
