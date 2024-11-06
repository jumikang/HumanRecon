import yaml
from dataset.loader_diffusion import create_dataset
from model import DiffusionModel


if __name__ == "__main__":
    # load a configuration file.
    path2config = './config/base_config_diffusion.yaml'
    with open(path2config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    train_dataloader = create_dataset(params['data'], validation=False)

    # train a model.
    model = DiffusionModel()
    model.train(train_dataloader, params=params['train'])
