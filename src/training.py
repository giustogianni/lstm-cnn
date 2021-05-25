import torch
import argparse

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--data_dir',
                    action='store_true', default=False,
                    help='Data location (default \'./data\')')

parser.add_argument('--model', type=str, required=True)         #model architecture, either "SRCNN" or "AE"
parser.add_argument('--lr', type=float, default=1e-4)           #learning rate
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-epochs', type=int, default=300)      #number of training epochs
parser.add_argument('--num-workers', type=int, default=8)

parser.add_argument('--seed',
                    type=int, default=0,
                    help='Radom seed (default 0, < 0 is no seeding)')

args = parser.parse_args()

if args.seed >= 0:
    torch.manual_seed(args.seed)