# experiments/si_split_cifar10.py

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import avalanche as avl
from avalanche.training import SynapticIntelligence
from avalanche.evaluation import metrics as metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
# from torchvision.models import resnet18
from experiments.utils import set_seed, create_default_args
from models import SimpleCNN


def si_split_cifar10(override_args=None):
    """
    Train Synaptic Intelligence (SI) on Split CIFAR-10 benchmark.
    """

    # Default args
    args = create_default_args({
        'lr': 0.001,
        'epochs': 2,          # small for testing
        'train_mb_size': 64,
        'eval_mb_size': 128,
        'si_lambda': 1.0,     # regularization strength
        'si_eps': 0.1,        # damping
        'seed': 0
    }, override_args)

    # Reproducibility
    set_seed(args.seed)

    # Benchmark: Split CIFAR-10 into 5 tasks (2 classes each)
    benchmark = avl.benchmarks.SplitCIFAR10(n_experiences=5, return_task_id=True)

    # Model: SimpleCNN for CIFAR-10
    model = SimpleCNN(num_classes=10)

    # Loss + Optimizer
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr)

    # Loggers + Evaluation
    interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open('si_split_cifar10_log.txt', 'w'))
    eval_plugin = EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger]
    )

    # Strategy: SI
    cl_strategy = SynapticIntelligence(
        model, optimizer, criterion,
        si_lambda=args.si_lambda,
        eps=args.si_eps,
        train_mb_size=args.train_mb_size,
        train_epochs=args.epochs,
        eval_mb_size=args.eval_mb_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        evaluator=eval_plugin
    )

    # Training loop
    results = []
    for experience in benchmark.train_stream:
        print("Start training on experience", experience.current_experience)
        cl_strategy.train(experience)
        print("End training, evaluating...")
        results.append(cl_strategy.eval(benchmark.test_stream[:experience.current_experience+1]))

    return results
