import os
import time
import datetime
import math
import cv2
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
import albumentations as albu
from albumentations import pytorch as albu_pytorch
from sklearn.model_selection import train_test_split
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
PATH_TO_DF = 'gdrive/My Drive/Colab Notebooks/Kaggle_Severstal/severstal-steel-data/train_sgm.csv'
LOG_DIR = 'gdrive/My Drive/Colab Notebooks/Kaggle_Severstal/log/segmentation'
DATA_DIR = '.'
LEARNING_RATE = 1e-4
MODEL_NAME = ''
EARLY_STOPPING = None
RANDOM_SEED = 42

# utils/utils.py
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(1600, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=np.int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], np.uint8)
    for st, en in zip(starts, ends):
        img[st:en] = 255

    return img.reshape(shape).T


def make_mask(row_idx, df):
    fname = df.iloc[row_idx].name
    labels = df.iloc[row_idx][:4].values

    mask = np.zeros((256, 1600, 4), dtype=np.uint8)
    for i, label in enumerate(labels):
        if not pd.isna(label):
            mask[:, :, i] = rle2mask(label)
    return fname, mask


def show_mask(row_idx, df, data_dir='severstal-steel-data/train_images',
              pallet=((250, 230, 20), (30, 200, 241), (200, 30, 250), (250, 60, 20)),
              contour=True, show=False):
    image_name, mask = make_mask(row_idx, df)
    img = cv2.imread(os.path.join(data_dir, image_name))

    if contour:
        for ch in range(4):
            contours, _ = cv2.findContours(mask[:, :, ch],
                            cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for i in range(0, len(contours)):
                cv2.polylines(img, contours[i], True, pallet[ch], 3)
    else:
        for ch in range(4):
            img[mask[:, :, ch] == 1] = pallet[ch]
    plt.imshow(img)

    if show:
        plt.show()


# utils/datasets.py
class SteelSegmentationDataset(Dataset):
    def __init__(self, df, data_dir='severstal-steel-data/train_images', phase='train', mean=None, std=None):
        self.df = df
        self.data_dir = data_dir
        self.transforms = get_transforms(phase, mean, std)

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image = cv2.imread(os.path.join(self.data_dir, image_id))

        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask'][0].permute(2, 0, 1)

        return image, mask

    def __len__(self):
        return len(self.df.index)


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
def dice_single_channel(targets, preds, eps=1e-9):
    batch_size = preds.shape[0]

    preds = preds.view((batch_size, -1)).float()
    targets = targets.view((batch_size, -1)).float()

    dice = (2 * (preds * targets).sum(1) + eps) / (preds.sum(1) + targets.sum(1) + eps)
    return dice


def mean_dice_score(targets, outputs, threshold=0.5):
    batch_size = outputs.shape[0]
    n_channels = outputs.shape[1]
    preds = (outputs.sigmoid() > threshold).float()

    mean_dice = 0
    for i in range(n_channels):
        dice = dice_single_channel(targets[:, i, :, :], preds[:, i, :, :])
        mean_dice += dice.sum(0) / (n_channels * batch_size)
    return mean_dice.item()


def pixel_accuracy_score(targets, outputs, threshold=0.5):
    preds = (outputs.sigmoid() > threshold).float()
    correct = torch.sum((targets == preds)).item()
    total = outputs.numel()
    return correct / total

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

# utils/optimizers.py
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


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
        self.criterion = criterion(pow_weight=torch.tensor([2.0, 2.0, 1.0, 1.5]))
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
        self.metrics_header = {'Dice': mean_dice_score, 'Pixelwise accuracy': pixel_accuracy_score}
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

        del images, targets, outputs, loss
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
os.makedirs(os.path.join(LOG_DIR, SESSION_ID))
os.makedirs(os.path.join(LOG_DIR, SESSION_ID, 'model_weights'))

df = pd.read_csv(PATH_TO_DF, index_col=0)
df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
df['NumDefects'] = df.count(axis=1)

model = smp.FPN(encoder_name='efficientnet-b2', encoder_weights='imagenet', classes=4, activation=None)
model_trainer = Trainer(model=model,
                        n_epochs=2,
                        batch_size={'train': 2, 'val': 2},
                        criterion=nn.BCEWithLogitsLoss,
                        optimizer=RAdam,
                        df=df,
                        dataset=SteelSegmentationDataset,
                        data_dir=DATA_DIR,
                        stratify_by=''
                        )
model_trainer.train()