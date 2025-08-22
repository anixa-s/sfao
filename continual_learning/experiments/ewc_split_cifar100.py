import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
# from torchvision.models import resnet18
from models import SimpleCNN

from experiments.utils import set_seed, create_default_args

def ewc_split_cifar100(override_args={'epochs': 2}):
    args = create_default_args({
        'cuda': 0,
        'ewc_lambda': 1,
        'epochs': 2, 'dropout': 0,
        'ewc_mode': 'separate', 'ewc_decay': None,
        'learning_rate': 0.001, 'train_mb_size': 128, 'seed': None
    }, override_args)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitCIFAR100(n_experiences=10, return_task_id=True)

    # Model: SimpleCNN for CIFAR-100
    model = SimpleCNN(num_classes=100)

    criterion = CrossEntropyLoss()
    interactive_logger = avl.logging.InteractiveLogger()
    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger]
    )

    cl_strategy = avl.training.EWC(
        model, SGD(model.parameters(), lr=args.learning_rate), criterion,
        ewc_lambda=args.ewc_lambda, mode=args.ewc_mode, decay_factor=args.ewc_decay,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res

if __name__ == '__main__':
    res = ewc_split_cifar100({'epochs': 2})
    print(res)