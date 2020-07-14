import numpy as np


class Meter:
    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.metrics_values = {m_name: [] for m_name in metrics.keys()}

    def compute(self, outputs, targets):
        for m_name, m_func in self.metrics:
            m_value = m_func(outputs, targets)
            self.metrics_values[m_name].append(m_value)

    def get_epoch_metrics(self):
        epoch_metrics = {}
        for m_name, m_value in self.metrics_values:
            epoch_metrics[m_name] = np.mean(m_value)

        return epoch_metrics

