import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set transformer detector', add_help=False)
    parser.add_argument('--network', default='highwaynet', type=str, help="Network type: method or highwaynet")
    parser.add_argument('--dataset', default='ngsim', type=str, help="Dataset type: ngsim or highd")

    # Socail Attention Mechanism
    parser.add_argument('--attn_nhead', default=4, type=int, help="Number of attention heads inside Social Attention Mechanism")
    parser.add_argument('--attn_out', default=16, type=int, help="Output dimension of the Social Attention Mechanism")

    # * Transformer parameters
    parser.add_argument('--input_dim', default=6, type=int, help="Dimension of the input features")
    parser.add_argument('--encoder_input_dim', default=64, type=int, help="Dimension of the encoder input features")
    parser.add_argument('--decoder_input_dim', default=128, type=int, help="Dimension of the decoder input features")
    parser.add_argument('--lat_dim', default=3, type=int, help="Number of classes for the classifier")
    parser.add_argument('--lon_dim', default=3, type=int, help="Number of classes for the classifier")
    parser.add_argument('--hist_length', default=16, type=int, help="The length of hisitory trajectory")
    parser.add_argument('--pred_length', default=25, type=int, help="Number of classes for the classifier")
    parser.add_argument('--enc_layers', default=2, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=128, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=9, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--activation', default='relu', type=str, help="Activation function in the transformer encoder: relu or None")
    parser.add_argument('--return_intermediate', action='store_false')
    
    # MMoE parameters
    parser.add_argument('--expert_output_size', default=128, type=int, help="Output size of the expert")

    # * Convolutional social parameters
    parser.add_argument('--soc_conv_dim', default=64, type=int, help="Dimension of the input in the social convolution")
    parser.add_argument('--soc_conv_depth', default=64, type=int, help="Number of channels in the social convolution")
    parser.add_argument('--conv_3x1_depth', default=16, type=int, help="Number of channels in the 3x1 convolutions")
    parser.add_argument('--grid_size', default=(13, 3), type=int, help="Size of the social context grid")

    # * Train
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://', type=str)

    return parser
