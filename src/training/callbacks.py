class Callback:
    def on_epoch_end(self, epoch, logs=None):
        pass

class EarlyStopping(Callback):
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss') if logs else None
        if current_loss is not None:
            if current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.stop_training = True
                    print(f"Early stopping at epoch {epoch+1}") 