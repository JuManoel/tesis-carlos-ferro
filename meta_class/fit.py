import math
from auxiliares.save_load_model import save_model
from auxiliares.save_load_model import load_model
from auxiliares.EarlyStopping import EarlyStopping
from auxiliares.metricas import calculate_metrics

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from lion_pytorch import Lion
from sklearn.model_selection import StratifiedKFold

from copy import deepcopy
import numpy as np

def fit_and_evaluate(model, t_x, t_y, v_x, v_y, optimizer, EPOCHS=500, batch_size=8, name="", scheduler=None):
    device = next(model.parameters()).device
    fn_loss = nn.BCEWithLogitsLoss(torch.tensor(274/226).to(device))
    early_stopping = EarlyStopping(patience=20, verbose=True)
    best_model_path = f'./models_class/{name}.pth'
    train_loader = data.DataLoader(data.TensorDataset(t_x, t_y), batch_size=batch_size, shuffle=True)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        all_train_labels = []
        all_train_outputs = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.to(device)
            loss = fn_loss(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            all_train_labels.extend(labels.detach().to("cpu").numpy())
            all_train_outputs.extend(outputs.detach().to("cpu").numpy())
            del inputs, labels, outputs
            torch.cuda.empty_cache()
        train_loss /= len(train_loader)
        train_metrics = calculate_metrics(all_train_labels, all_train_outputs)
        print(f"name: {name} | Epoch {epoch+1}/{EPOCHS}: \n"
              f"Train(loss: {train_loss:.4f}, Metrics: {train_metrics})")
        with torch.no_grad():
            metricas = predict(model, v_x, v_y, batch_size)
        print(f'Validacion(loss: {metricas["loss"]}, Metrics: {metricas["metrics"]}')
        early_stopping(metricas["metrics"]["accuracy"])
        if(math.isnan(train_loss)):
            break
        if early_stopping.counter == 0:
            save_model(model, best_model_path)
        if scheduler:
                scheduler.step(metrics=metricas["metrics"]["accuracy"])
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        print("-" * 126)
    del train_loader, all_train_labels, all_train_outputs, train_metrics, metricas
    del t_x, t_y, v_x, v_y
    torch.cuda.empty_cache()
    return early_stopping.best_metric

def to_GPU(model):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Nenhuma GPU disponível. Usando a CPU.")
        device = torch.device("cpu")
        model.to(device)
        return device

    print(f"GPUs disponíveis: {num_gpus}")
    
    for gpu_id in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(gpu_id)
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9  # Em GB
        allocated_memory = torch.cuda.memory_reserved(gpu_id) / 1e9  # Em GB
        free_memory = total_memory - allocated_memory
        
        print(f"GPU {gpu_id} ({gpu_name}):")
        print(f"  Memória total: {total_memory:.2f} GB")
        print(f"  Memória alocada: {allocated_memory:.2f} GB")
        print(f"  Memória livre: {free_memory:.2f} GB")
        
        if free_memory >= 20.0:
            print(f"Usando GPU {gpu_id} ({gpu_name}) com {free_memory:.2f} GB de memória livre.")
            device = torch.device(f"cuda:{gpu_id}")
            model.to(device).float()  # Converter para meia precisão
            return device
        
def selectOptimizer(model, optimizer_name, lr, weight_decay):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=lr,
            betas=(0.98, 0.92),
            eps=1e-8,
            weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=lr,
            betas=(0.98, 0.92),
            eps=1e-8,
            weight_decay=weight_decay
        )
    elif optimizer_name == "Lion":
        optimizer = Lion(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Otimizador {optimizer_name} não reconhecido.")
    return optimizer

def train(model, X, Y, optimizer_name, epochs=500, batch_size=8, num_splits=5, name=""):
    model_copy = deepcopy(model)
    model_copy.float()
    best_metric = -1
    kfold = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_recall = []
    fold_precision = []
    fold_f1 = []
    changed_gpu = True
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, Y)): 
        print(f"\nTreinando Fold {fold + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        X_train = torch.Tensor(X_train).float()
        y_train = torch.tensor(y_train, dtype=torch.uint8)
        torch.cuda.empty_cache()
        while True:
            try:
                device = to_GPU(model_copy)
                break
            except RuntimeError as e: 
                if 'out of memory' in str(e):
                   torch.cuda.empty_cache()
                   device = to_GPU(model_copy)
                else:
                    raise e
        while True:
            try:
                # Aumentar 15% do dataset de treino com ruídos
                optimizer = selectOptimizer(model_copy, optimizer_name, 1e-4, 1e-3)
                scheduler = ReduceLROnPlateau(
                    optimizer=optimizer,     # Passa o otimizador
                    mode='max',         # Minimiza a métrica monitorada
                    factor=0.5,         # Reduz a lr pela metade
                    patience=5,         # Espera 5 épocas sem melhora antes de reduzir
                    verbose=True,       # Mostra logs
                    min_lr=1e-6         # Limite inferior para a lr
                )
                metric = fit_and_evaluate(model_copy, X_train, y_train, X_val, y_val, optimizer, epochs, batch_size, name=name, scheduler=scheduler)
                break
            except RuntimeError as e: #sirve para reducir el batch_size, caso la memoria no alcansa
                if 'out of memory' in str(e) or 'CUBLAS_STATUS_ALLOC_FAILED' in str(e):
                    torch.cuda.empty_cache()
                    if changed_gpu:
                        print(f"Reducing batch size from {batch_size} to {batch_size // 2}")
                        batch_size = batch_size // 2
                        if batch_size == 0:
                            raise RuntimeError("Batch size reduced to 0. Cannot proceed further.")
                        changed_gpu = False
                    else:
                        device = to_GPU(model_copy)
                        changed_gpu = True
                else:
                    raise e
        del X_train, y_train
        torch.cuda.empty_cache()
        model_copy = load_model(f'./models_class/{name}.pth', model_copy)
        with torch.no_grad():
            metricas = predict(model_copy, X_val, y_val, batch_size)
        with open("results_meta.txt", "a") as f: #guardo las prediciones y el true label para cada fold
            f.write("name: " + name + f"Fold {fold + 1}""\n")
            f.write("y_test: " + str(y_val) + "\n")
            f.write("y_pred: " + str(metricas["predictions"]) + "\n") # numeros + es para classe 1 y - classe 0
        del X_val, y_val
        torch.cuda.empty_cache()
        print(f"Metricas Validacion Fold {fold + 1}: ", metricas["metrics"])
        fold_accuracies.append(metricas["metrics"]["accuracy"])
        fold_recall.append(metricas["metrics"]["recall"])
        fold_precision.append(metricas["metrics"]["precision"])
        fold_f1.append(metricas["metrics"]["f1_score"])
        if(metric > best_metric):
            best_metric = metric
            save_model(model_copy, f'./models_class/{name}.pth')
        del model_copy, scheduler
        torch.cuda.empty_cache()
        model_copy = deepcopy(model).to(device).float()        
        print("=" * 126)
    mean_accuracy = np.mean(fold_accuracies)
    mean_recall = np.mean(fold_recall)
    mean_precision = np.mean(fold_precision)
    mean_f1 = np.mean(fold_f1)
    std_accuracy = np.std(fold_accuracies)
    std_recall = np.std(fold_recall)
    std_precision = np.std(fold_precision)
    std_f1 = np.std(fold_f1)
    with open("results_meta.txt", "a") as f: #guarda las metricas de validadcion para cada fold
            f.write("name: " + name +"\n")
            f.write(f"\nAcurácia Média: {mean_accuracy:.4f} ± {std_accuracy:.4f}" + "\n")
            f.write(f"\nRecall Média: {mean_recall:.4f} ± {std_recall:.4f}" + "\n")
            f.write(f"\nPrecicion Média: {mean_precision:.4f} ± {std_precision:.4f}" + "\n")
            f.write(f"\nF1 Média: {mean_f1:.4f} ± {std_f1:.4f}" + "\n")
    print(f"Acurácia Média: {mean_accuracy:.4f} ± {std_accuracy:.4f}"+
          f"\nPrecicion Média: {mean_precision:.4f} ± {std_precision:.4f}"+
          f"\nRecall Média: {mean_recall:.4f} ± {std_recall:.4f}"+
          f"\nF1 Média: {mean_f1:.4f} ± {std_f1:.4f}")
    model = load_model(f'./models_class/{name}.pth', model)
    model.to("cpu")
    del model_copy
    del X, Y
    torch.cuda.empty_cache()
    return model

def predict(model, X, Y, batch_size=32):
    device = next(model.parameters()).device
    fn_loss = nn.BCEWithLogitsLoss(torch.tensor(274/226).to(device))
    X = torch.Tensor(X).float()
    Y = torch.tensor(Y, dtype=torch.uint8)
    loader = data.DataLoader(data.TensorDataset(X, Y), batch_size=batch_size, shuffle=False)    
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = fn_loss(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item()
            all_preds.append(outputs.detach().to("cpu").numpy())
            all_labels.extend(labels.detach().to("cpu").numpy())
            del inputs, labels, outputs
            torch.cuda.empty_cache()
    val_loss /= len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    val_metrics = calculate_metrics(all_labels, all_preds)
    del loader, all_labels
    torch.cuda.empty_cache()
    return {
        "predictions": all_preds,
        "loss": val_loss,
        "metrics": val_metrics
    }