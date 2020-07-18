import os
import time
import datetime
import cv2
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
from albumentations import pytorch as albu_pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as roc_auc_skl
from torchvision.models import resnet34
import warnings
warnings.filterwarnings('ignore')


# JS functions to prevent colab runtime disconnect
'''function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);'''

# configs/train_params.py
SESSION_ID = datetime.datetime.now().strftime('%y.%m.%d_%H-%M')
PATH_TO_DF = 'gdrive/My Drive/Colab Notebooks/Kaggle_Severstal/severstal-steel-data/train_clf.csv'
LOG_DIR = 'gdrive/My Drive/Colab Notebooks/Kaggle_Severstal/log'
DATA_DIR = '.'
LEARNING_RATE = 1e-4
MODEL_NAME = 'efficient_net_b1'
EARLY_STOPPING = None
RANDOM_SEED = 42

# utils/utils.py


# utils/datasets.py
class SteelClassificationDataset(Dataset):
    def __init__(self, df, data_dir='severstal-steel-data/train_images', phase='train', mean=None, std=None):
        self.df = df
        self.data_dir = data_dir
        self.transforms = get_transforms(phase, mean, std)

    def __getitem__(self, idx):
        fname, target = self.df.iloc[idx].values
        image = cv2.imread(os.path.join(self.data_dir, fname))
        image = self.transforms(image=image)['image']

        return image, float(target)

    def __len__(self):
        return len(self.df)


def get_transforms(phase, mean, std, list_transforms=None):
    if not list_transforms:
        list_transforms = []

    if phase == 'train':
        list_transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomBrightnessContrast(p=0.1, brightness_limit=0.1, contrast_limit=0.1)
            ]
        )
    list_transforms.extend(
        [
            albu.Normalize(mean=mean, std=std),
            albu_pytorch.ToTensor()
        ]
    )

    list_transforms = albu.Compose(list_transforms)
    return list_transforms


def data_provider(df, data_dir, phase, dataset_cls, batch_size, stratify_by, n_workers, mean, std):
    train_df, val_df = train_test_split(df, test_size=.2, stratify=df[stratify_by], random_state=RANDOM_SEED)
    df = train_df if phase == 'train' else val_df

    dataset = dataset_cls(df, data_dir, phase, mean, std)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            pin_memory=False,
                            shuffle=True
                            )
    return dataloader


# utils/metrics.py
def roc_auc_score(targets, outputs):
    probs = torch.sigmoid(outputs)
    return roc_auc_skl(targets, probs)


def accuracy_score(targets, outputs, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()
    return (targets == preds).float().mean().item()


# utils/meter.py
class Meter:
    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.metrics_values = {m_name: [] for m_name in metrics.keys()}

    def compute(self, outputs, targets):
        for m_name, m_func in self.metrics.items():
            m_value = m_func(targets, outputs)
            self.metrics_values[m_name].append(m_value)

    def get_epoch_metrics(self):
        epoch_metrics = {}
        for m_name, m_value in self.metrics_values.items():
            epoch_metrics[m_name] = np.mean(m_value)

        return epoch_metrics


# utils/logger.py
class Logger:
    def __init__(self, name, session_id='', format='%(message)s'):
        self.name = name
        self.format = format
        self.level = logging.INFO
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.path_to_log = os.path.join(LOG_DIR, session_id, name + '.log')
        self.verbose = False

        # Logger configuration
        self.formatter = logging.Formatter(self.format)
        self.file_handler = logging.FileHandler(self.path_to_log)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

        if self.verbose:
            self.stream_handler = logging.StreamHandler()
            self.stream_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.stream_handler)

        self.date_format = '%Y-%m-%d %H:%M'

    def info(self, msg):
        self.logger.info(msg)

    def start_log(self, comment):
        msg = 'Training started at ' + datetime.datetime.now().strftime('%H:%M:%S, %b %d') + '\n'
        msg += comment + '\n'
        msg += '___________________________________________________________________\n'
        self.info(msg)

    def epoch_log(self, epoch, phases, losses, metrics):
        msg = 'Epoch №' + str(epoch) + ' passed at ' + datetime.datetime.now().strftime('%H:%M') + '\n'
        for phase in phases:
            msg += (phase.capitalize() + ' loss: ').ljust(12) + str(round(losses[phase][-1], 7)) + '\n'

            msg += (phase.capitalize() + ' metrics').ljust(13) + ' ---> '
            for m_name, m_value in metrics[phase].items():
                msg += m_name.capitalize() + ': ' + str(round(m_value[-1], 5)) + '\t'
            msg += '\n'

        self.info(msg)


# utils/trainer.py
class Trainer:
    def __init__(self, model, n_epochs=10, batch_size={'train': 8, 'val': 8},
                 criterion=nn.BCEWithLogitsLoss, optimizer=optim.Adam,
                 df=None, dataset=None, stratify_by='',
                 data_dir='', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.accumulation_steps = 256 // batch_size['train']
        self.phases = ['train', 'val']
        self.device = torch.device('cuda:0')#cpu')
        self.model = model
        self.model.to(self.device)
        self.criterion = criterion()
        self.optimizer = optimizer(model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.9, mode="min", patience=3, verbose=True)
        self.dataloaders = {
            phase: data_provider(df=df,
                                 data_dir=data_dir,
                                 phase=phase,
                                 dataset_cls=dataset,
                                 batch_size=self.batch_size[phase],
                                 stratify_by=stratify_by,
                                 n_workers=0,
                                 mean=mean, std=std)
            for phase in self.phases
        }
        self.metrics_header = {'Accuracy': accuracy_score, 'ROC-AUC': roc_auc_score}
        self.losses = {phase: [] for phase in self.phases}
        self.metrics = {
            phase: {m_name: [] for m_name in self.metrics_header}
            for phase in self.phases
        }
        self.best_scores = np.array([-np.inf for _ in self.metrics_header.keys()])
        self.lr = LEARNING_RATE

        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        # torch.set_default_tensor_type("torch.FloatTensor")
        torch.backends.cudnn.benchmark = True

    def _train_step(self, epoch, phase):
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")

        meter = Meter(metrics=self.metrics_header)
        self.model.train(phase == "train")
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        # tk0 = tqdm(dataloader, total=total_batches)

        self.optimizer.zero_grad()
        for i, (images, targets) in enumerate(dataloader):
            images, targets = torch.tensor(images, dtype=torch.float), torch.tensor(targets, dtype=torch.float)
            images, targets = images.to(self.device), targets.to(self.device)
            targets = targets.unsqueeze(1)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss = loss / self.accumulation_steps

            if phase == "train":
                loss.backward()
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()

            outputs = outputs.detach().cpu()
            targets = targets.detach().cpu()
            try:
                meter.compute(outputs, targets)
            except:
                running_loss -= loss.item()
                total_batches -= 1

            # tk0.update(1)
            # tk0.set_postfix(loss=(running_loss / (i + 1)))
        # tk0.close()
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        self.losses[phase].append(epoch_loss)
        epoch_metrics = meter.get_epoch_metrics()
        for m_name in self.metrics_header:
            self.metrics[phase][m_name].append(epoch_metrics[m_name])

        torch.cuda.empty_cache()
        return epoch_loss, epoch_metrics

    def train(self):
        logger = Logger(name='logger', session_id=SESSION_ID)
        logger.start_log('')

        epochs_wo_improve = 0
        for epoch in range(self.n_epochs):
            if EARLY_STOPPING is not None and epochs_wo_improve >= EARLY_STOPPING:
                print('Early stopping {}'.format(EARLY_STOPPING))
                torch.save(state, "{}/model_weights/model_{}_epoch_{}_score_{}.pth".format(
                    os.path.join(LOG_DIR, SESSION_ID), MODEL_NAME, epoch, scores[0]))
                break

            self._train_step(epoch, 'train')
            state = {
                "epoch": epoch,
                "best_metric": self.best_scores,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }

            val_loss, val_metrics = self._train_step(epoch, 'val')
            print(f'Val loss: {val_loss}\n{val_metrics}\n------------------------------------------------------|')
            logger.epoch_log(epoch, self.phases, self.losses, self.metrics)
            self.scheduler.step(val_loss)

            scores = np.fromiter(val_metrics.values(), dtype=np.float)
            if (scores > self.best_scores).all():
                print('*****New optimal model found and saved*****')
                state['best_metric'] = val_metrics
                torch.save(state, "{}/model_weights/model_{}_epoch_{}_score_{}.pth".format(
                    os.path.join(LOG_DIR, SESSION_ID), MODEL_NAME, epoch, scores[0]))

                self.best_scores = scores
            else:
                epochs_wo_improve -= 1


# train.py
os.makedirs(os.path.join(LOG_DIR, SESSION_ID), exist_ok=False)
os.makedirs(os.path.join(LOG_DIR, SESSION_ID, 'model_weights'))
df = pd.read_csv(PATH_TO_DF, index_col=0)
model = resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
model_trainer = Trainer(model=model,
                        n_epochs=2,
                        batch_size={'train': 2, 'val': 2},
                        criterion=nn.BCEWithLogitsLoss,
                        optimizer=optim.Adam,
                        df=df,
                        dataset=SteelClassificationDataset,
                        data_dir='severstal-steel-data/train_images',
                        stratify_by='HasDefect'
                        )
model_trainer.train()