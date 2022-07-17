import logging
import torch
import action
from parameter_parser import parameter_parser


def config_logger():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(ch)


def main(args):
    if args.action == 'model_train':
        if args.unlearning_method == 'scratch':
            action.ActionModelTrainScratch(args)
        else:
            action.ActionModelTrainSisa(args)
    elif args.action == 'attack':
        if args.unlearning_method == 'scratch':
            action.ActionAttackScratch(args)
        else:
            action.ActionAttackSisa(args)
    else:
        raise Exception(f'Invalid action: No {args.action}')


if __name__ == '__main__':
    args = parameter_parser().parse_args()
    config_logger()
    main(args)

