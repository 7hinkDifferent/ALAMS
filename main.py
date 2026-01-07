import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
import datetime
import os, json
import copy

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--full_init', action='store_true', default=False, help="use full init labeled samples")
    parser.add_argument('--n_init_labeled', type=int, default=5000, help="number of init labeled samples") # original 10000
    parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
    parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
    parser.add_argument('--dataset_name', type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
    parser.add_argument('--strategy_name', type=str, default="RandomSampling",
                        choices=["RandomSampling",
                                 "LeastConfidence",
                                 "MarginSampling",
                                 "EntropySampling",
                                 "LeastConfidenceDropout",
                                 "MarginSamplingDropout",
                                 "EntropySamplingDropout",
                                 "KMeansSampling",
                                 "KCenterGreedy",
                                 "BALDDropout",
                                 "AdversarialBIM",
                                 "AdversarialDeepFool",
                                 "Sequential",
                                 "WeightedSumGaussian",
                                 "WeightedSumGMM",
                                 "WeightedGeometricMeanGaussian",
                                 "WeightedGeometricMeanGMM",
                                 "MultiArmBandit",
                                 "AdaptiveMixtureStrategiesGaussian",
                                 "AdaptiveMixtureStrategiesGMM"], help="query strategy")
    args = parser.parse_args()
    print(vars(args))
    print()

    # fix random seed
    set_seed(args.seed)
    # logging
    strategy_name = "full" if args.full_init else args.strategy_name
    exp_name = f'{args.dataset_name}_{strategy_name}_seed{args.seed}_{datetime.datetime.now().strftime("%H%M%S")}'
    logging_root = f'./logs/{datetime.datetime.now().strftime("%Y%m%d")}/{exp_name}'
    os.makedirs(logging_root, exist_ok=True)
    new_labeled_idxs = {}
    metrics = {}
    data_stat = {}

    print(f'Logging to: {logging_root}\n')

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using device: {device}\n')

    dataset = get_dataset(args.dataset_name)                   # load dataset
    net = get_net(args.dataset_name, device, logging_root=logging_root)                   # load network
    strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy


    # start experiment
    init_labeled_idxs = dataset.initialize_labels(args.n_init_labeled, full_init=args.full_init)
    data_stat['round_0'] = dataset.get_statistics()
    new_labeled_idxs["round_0"] = {"index": init_labeled_idxs.tolist()}
    print(f"number of labeled pool: {args.n_init_labeled if not args.full_init else dataset.n_pool}")
    print(f"number of unlabeled pool: {dataset.n_pool - (args.n_init_labeled if not args.full_init else dataset.n_pool)}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    # round 0 accuracy
    print("Round 0")
    train_results = strategy.train(round=0)
    preds, acc, loss = strategy.predict(dataset.get_test_data())
    accuracy = dataset.cal_test_acc(preds)
    print(f"Round 0 testing accuracy: {accuracy}")
    metrics["round_0"] = {'test_accuracy': accuracy, 'test_loss': loss}

    if not args.full_init:

        for rd in range(1, args.n_round+1):
            print(f"Round {rd}")

            # feedback with train results
            feedback_meta = strategy.feedback(train_results)

            # query
            query_idxs, query_meta = strategy.query(args.n_query)
            new_labeled_idxs[f"round_{rd}"] = {
                "index": query_idxs.tolist(),
                "feedback": copy.deepcopy(feedback_meta) or {},
                "query": copy.deepcopy(query_meta) or {},
            }

            # update labels
            strategy.update(query_idxs)
            data_stat[f"round_{rd}"] = dataset.get_statistics()
            # retrain the model
            strategy.reset_net()
            train_results = strategy.train(round=rd)

            # calculate accuracy
            preds, acc, loss = strategy.predict(dataset.get_test_data())
            accuracy = dataset.cal_test_acc(preds)
            print(f"Round {rd} testing accuracy: {accuracy}")
            metrics[f"round_{rd}"] = {'test_accuracy': accuracy, 'test_loss': loss}

    # save results
    try:
        with open(f'{logging_root}/new_labeled_idxs.json', 'w') as f:
            json.dump(new_labeled_idxs, f)
        with open(f'{logging_root}/data_stat.json', 'w') as f:
            json.dump(data_stat, f)
        with open(f'{logging_root}/metrics.json', 'w') as f:
            json.dump(metrics, f)
    except Exception as e:
        print(f"Error saving results: {e}")
        # save with txt
        with open(f'{logging_root}/new_labeled_idxs.txt', 'w') as f:
            f.write(str(new_labeled_idxs))
        with open(f'{logging_root}/data_stat.txt', 'w') as f:
            f.write(str(data_stat))
        with open(f'{logging_root}/metrics.txt', 'w') as f:
            f.write(str(metrics))


if __name__ == '__main__':
    # On macOS and Windows, multiprocessing uses 'spawn'.
    # Protect main to avoid re-executing top-level code in worker processes.
    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # start method already set; safe to proceed
        pass
    main()
