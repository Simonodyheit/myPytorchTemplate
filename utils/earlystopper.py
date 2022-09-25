"""
# Created in 2021/10/28
# reference: SSLMRIRecon
"""
class EarlyStopping:
    def __init__(self, patience=15, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop_flag = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        score = -1.0 *  metrics if loss else metrics
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_flag = True
        else:
            self.best_score = score
            self.counter = 0

if __name__ == '__main__':
    early_stopper = EarlyStopping()
    pass
