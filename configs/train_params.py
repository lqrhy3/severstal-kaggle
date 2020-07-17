import datetime
import os

SESSION_ID = datetime.datetime.now().strftime('%y.%m.%d_%H-%M')
# PATH_TO_DF = 'data/train_clf.csv'
PATH_TO_DF = 'data/debug_clf.csv'
LOG_DIR = os.path.join('log')
LEARNING_RATE = 0.0001
MODEL_NAME = 'resnet34'
EARLY_STOPPING = None
RANDOM_SEED = 42
