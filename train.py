import pandas as pd
import torch.optim as optim
import torch.nn as nn
from utils.trainer import Trainer
from utils.datasets import SteelClassificationDataset
from configs.train_params import *
from torchvision.models import resnet34
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    os.makedirs(os.path.join(LOG_DIR, SESSION_ID), exist_ok=False)
    os.makedirs(os.path.join(LOG_DIR, SESSION_ID, 'model_weights'))
    df = pd.read_csv(PATH_TO_DF, index_col=0)
    model = resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model_trainer = Trainer(model=model,
                            n_epochs=1,
                            batch_size={'train': 1, 'val': 2},
                            criterion=nn.BCEWithLogitsLoss,
                            optimizer=optim.Adam,
                            df=df,
                            dataset=SteelClassificationDataset,
                            data_dir='data/debug_images',
                            stratify_by='HasDefect'
                            )
    model_trainer.train()
