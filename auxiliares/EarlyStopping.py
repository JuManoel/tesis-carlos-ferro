class EarlyStopping:
    """
    Classe por evaluar el modelo y detener el entrenamiento si el desempeño no mejora
    por defecto el espera 20 epocas
    """
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, metric):
        if self.best_metric is None:
            self.best_metric = metric
        elif metric < self.best_metric:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_metric = metric
            self.counter = 0
