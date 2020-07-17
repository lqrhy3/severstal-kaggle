import os
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from utils.datasets import data_provider
from utils.meter import Meter
from utils.logger import Logger
from utils.metrics import accuracy_score, roc_auc_score
from configs.train_params import *
import warnings
warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, model, n_epochs=10, batch_size={'train': 8, 'val': 8},
                 criterion=nn.BCEWithLogitsLoss, optimizer=optim.Adam,
                 df=None, dataset=None, stratify_by='',
                 data_dir='', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.accumulation_steps = 32 // batch_size['train']
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
                                 n_workers=4,
                                 mean=mean, std=std)
            for phase in self.phases
        }
        self.metrics_header = {'Accuracy': accuracy_score, 'ROC-AUC': accuracy_score}
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
        print('Train step started')
        print(f'Cuda memory allocated: {torch.cuda.memory_allocated()}'
              f'\nCuda memory cached: {torch.cuda.memory_cached()}')

        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")

        meter = Meter(metrics=self.metrics_header)
        self.model.train(phase == "train")
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        tk0 = tqdm(dataloader, total=total_batches)

        self.optimizer.zero_grad()
        for i, (images, targets) in enumerate(dataloader):
            images = torch.tensor(targets, dtype=torch.float).to(self.device)
            targets = torch.tensor(images, dtype=torch.float).to(self.device)
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

            tk0.update(1)
            tk0.set_postfix(loss=(running_loss / (i + 1)))
        tk0.close()
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        self.losses[phase].append(epoch_loss)
        epoch_metrics = meter.get_epoch_metrics()
        for m_name in self.metrics_header:
            self.metrics[phase][m_name].append(epoch_metrics[m_name])

        print('Train step finished')
        torch.cuda.empty_cache()
        print('Cache deleted')
        print(f'Cuda memory allocated: {torch.cuda.memory_allocated()}'
              f'\nCuda memory cached: {torch.cuda.memory_cached()}')
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
            logger.epoch_log(epoch, self.phases, self.losses, self.metrics)
            self.scheduler.step(val_loss)

            scores = np.fromiter(val_metrics.values(), dtype=np.float)
            if (scores > self.best_scores).all():
                print('*****New optimal model found and saved*****')
                state['best_metric'] = val_metrics
                torch.save(state, "{}/model_weights/model_{}_epoch_{}_score_{}.pth".format(
                    os.path.join(LOG_DIR, SESSION_ID), MODEL_NAME, epoch, scores[0]))

                self.best_scores = scores
            elif epoch % 5 == 0:
                print('Regular model save')
                torch.save(state, "{}/model_weights/model_{}_epoch_{}_score_{}.pth".format(
                    os.path.join(LOG_DIR, SESSION_ID), MODEL_NAME, epoch, scores[0]))
                epochs_wo_improve -= 1
            else:
                epochs_wo_improve -= 1
