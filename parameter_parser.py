import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parameter_parser():
    parser = argparse.ArgumentParser(description='LuZhaobo Toy Example')
    parser.add_argument('--action', type=str, default='attack',
                        choices=['model_train', 'attack'],
                        help="'mem_train' train the original-unlearning model pairs, 'attack' launch the attack")
    parser.add_argument('--dataset_ID', default=False, type=int, help='mnist=0, cifar10=1, stl10=2, cifar100=3')
    parser.add_argument('--datasets', nargs='+', default=['cifar10', 'cifar10', 'stl10', 'cifar100'])
    parser.add_argument('--dataset_name', type=str, default='mnist', choices=['mnist', 'cifar10', 'stl10', 'cifar100'])
    parser.add_argument('--num_classes', nargs='+', default=[10, 10, 10, 100])
    parser.add_argument('--is_sample', type=str2bool, default=True)
    parser.add_argument('--distance_type', type=int, default=0, help='0 is l0, 1 is l1, 2 is l2, 3 is linf')
    parser.add_argument('--is_train_multiprocess', type=str2bool, default=False)
    parser.add_argument('--Split-Size', nargs='+',
                        default=[[3000, 2000, 1500, 1000, 500, 100],  # 3000, 2000, 1500, 1000, 500, 100
                                 [7000, 6000, 5000, 4000, 3000, 2000],
                                 # 9000, 8000, 7000, 6000, 5000, 4000  # 7000, 6000, 5000, 4000, 3000, 2000
                                 [600, 500, 400, 300, 200, 100],  # 600, 500, 400, 300, 200, 100
                                 [350, 300, 250, 200, 150, 100],  # 350, 300, 250, 200, 150, 100
                                 ])
    parser.add_argument('--input_shape', default={'mnist': (1, 28, 28), 'cifar10': (3, 32, 32), 'cifar100': (3, 32, 32), 'stl10': (3, 32, 32)})
    parser.add_argument('--batch_size', nargs='+', default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--shadow-size', nargs='+', default=[46000, 42000, 38109, 1417],
                        help='size of four shadow dataset, is a list of length 4')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100), mnist is 50')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.05 for adam; 0.1 for SGD)')
    parser.add_argument('--optim', type=str, default="SGD", choices=['Adam', 'SGD'])
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--blackadvattack', default='HopSkipJump', type=str,
                        help='adversaryTwo uses the adv attack the target Model: HopSkipJump; QEBA')
    parser.add_argument('--logdir', type=str, default='./data/module and data/',
                        help='target log directory')
    parser.add_argument('--mode_type', type=str, default='',
                        help='the type of action referring to the load dataset')
    parser.add_argument('--advOne_metric', type=str, default='Loss_visual',
                        help='AUC of Loss, Entropy, Maximum respectively; or Loss_visual')
    parser.add_argument('--trainTargetModel', action='store_true',
                        help='Train a target model, if false then load an already trained model')
    parser.add_argument('--unlearning_method', type=str, default='scratch', choices=['scratch', 'sisa'])

    parser.add_argument('--shadow_set_num', type=int, default=10,
                        help="Number of shadow original model")
    parser.add_argument('--shadow_set_size', type=int, default=2000,
                        help="Number of shadow model training samples")
    parser.add_argument('--shadow_unlearning_size', type=int, default=20,
                        help="Number of unlearned model")
    parser.add_argument('--shadow_unlearning_num', type=int, default=1,
                        help="Number of deleted records to generate unlearned model")
    parser.add_argument('--shadow_num_shard', type=int, default=10,
                        help="Number of shards")

    parser.add_argument('--target_set_num', type=int, default=10,
                        help="Number of target original model")
    parser.add_argument('--target_set_size', type=int, default=2000,
                        help="Number of target model training samples")
    parser.add_argument('--target_unlearning_size', type=int, default=20,
                        help="Number of unlearned model")
    parser.add_argument('--target_unlearning_num', type=int, default=1,
                        help="Number of deleted records to generate unlearned model")
    parser.add_argument('--target_num_shard', type=int, default=10,
                        help="Number of shards")
    return parser
