import os
import datetime
import logging
import warnings
warnings.filterwarnings('ignore')


LOG_DIR = 'log'


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
