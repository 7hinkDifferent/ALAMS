from dataclasses import dataclass, field


@dataclass
class Config:
    # basic
    seed: int = 42
    device: str = 'cuda'  # or 'cpu'
    
    # logging
    logging_dir: str = './logs'

    # dataset
    data_dir: str = './data'
    dataset: str = 'MNIST'
    dataset_kwargs: dict = field(default_factory=lambda: {
        'batch_size': 64,
        'num_workers': 2
    })

    # model
    model : str = 'vgg16'
    model_kwargs: dict = field(default_factory=lambda: {
        'num_classes': 10
    })
    model_save_path: str = './model.pth'

    # optimizer
    optimizer: str = 'sgd'
    optimizer_kwargs: dict = field(default_factory=lambda: {
        'lr': 0.001,
        'momentum': 0.9
    })
    num_epochs: int = 10

    # active learning
    strategy: str = 'RandomSampling'
    n_init_labeled: int = 1000
    n_query: int = 1000
    n_round: int = 10

    @property
    def exp_name(self) -> str:
        ...

