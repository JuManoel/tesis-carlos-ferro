import math
from auxiliares.save_load_model import save_model
from auxiliares.save_load_model import load_model
from auxiliares.EarlyStopping import EarlyStopping
from auxiliares.metricas import calculate_metrics

import torch
import torch.nn as nn
import torch.utils.data as data
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms.functional import to_pil_image
import torch.optim as optim
from lion_pytorch import Lion

from monai.transforms import RandGaussianNoise
from monai.transforms import RandGibbsNoise
from monai.transforms import RandKSpaceSpikeNoise

from sklearn.model_selection import StratifiedKFold

from copy import deepcopy
import numpy as np

def fit_and_evaluate(model, t_x, t_y, v_x, v_y, optimizer, EPOCHS=500, batch_size=8, name="", scheduler=None):
    # funcion responsable por entrenar el modelo
    # y guardar el mejor modelo
    # t_x, t_y: datos de entrenamiento
    # v_x, v_y: datos de validacion
    device = next(model.parameters()).device
    # por esa funcion loss, no aplico sigmoid en el modelo
    fn_loss = nn.BCEWithLogitsLoss(torch.tensor(274/226).to(device)) # 274/226 es el peso de la clase 1
    early_stopping = EarlyStopping(patience=20, verbose=True)
    best_model_path = f'./models_class/{name}.pth'
    # genero el data loader para el entrenamiento
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
            # hago las prediciones
            metricas = predict(model, v_x, v_y, batch_size)
        print(f'Validacion(loss: {metricas["loss"]}, Metrics: {metricas["metrics"]}')
        early_stopping(metricas["metrics"]["accuracy"])
        if(math.isnan(train_loss)):
            # caso llegue al punto que no fue possible calcular el loss, que termine el entrenamiento
            break
        if early_stopping.counter == 0:
            # guardo el mejor modelo
            save_model(model, best_model_path)
        if scheduler:
                # aplico el scheduler
                scheduler.step(metrics=metricas["metrics"]["accuracy"])
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        print("-" * 126)
    del train_loader, all_train_labels, all_train_outputs, train_metrics, metricas
    del t_x, t_y, v_x, v_y
    torch.cuda.empty_cache()
    return early_stopping.best_metric

# funcion responsable por buscar 1 gpu con 20gb de memoria libre
# si no hay gpu disponible devuelve la cpu
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

# funcion que permite elejir un optimizador por el nombre
# recomiendo que para ese problema de @CarlosFerro, quede con 
# AdamW o Lion, ya que son los que mejor resultado dan
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



# funcion responsable por entrenar modelos con kfolds, guardar el mejor modelo y hacer predicciones

def train(model, X, Y, optimizer_name, epochs=500, batch_size=8, num_splits=5, name=""):
    model_copy = deepcopy(model) # copio el modelo
    model_copy.float()
    best_metric = -1
    # genero el kfolds
    kfold = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_recall = []
    fold_precision = []
    fold_f1 = []
    # aumento la data con ruido de monai
    rGaussian = RandGaussianNoise(prob=0.5, mean=0.0, std=0.1)
    rGibbs = RandGibbsNoise(prob=0.5, alpha=(0.6,0.8))
    rkspike = RandKSpaceSpikeNoise(prob=0.5, intensity_range=(11, 12))
    changed_gpu = True
    # empiezo el entrenamiento kfold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, Y)): 
        print(f"\nTreinando Fold {fold + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        X_train = torch.Tensor(X_train).float()
        y_train = torch.tensor(y_train, dtype=torch.uint8)
        
        num_samples = int(0.05 * len(X_train))
        indices = np.random.choice(len(X_train), num_samples, replace=False)
        X_train_aug = X_train[indices]
        y_train_aug_gaussian = y_train[indices]
        X_train_aug_gaussian = rGaussian(X_train_aug)

        num_samples = int(0.05 * len(X_train))
        indices = np.random.choice(len(X_train), num_samples, replace=False)
        X_train_aug = X_train[indices]
        y_train_aug_gibbs = y_train[indices]
        X_train_aug_gibbs = rGibbs(X_train_aug)

        num_samples = int(0.05 * len(X_train))
        indices = np.random.choice(len(X_train), num_samples, replace=False)
        X_train_aug = X_train[indices]
        y_train_aug_kspike = y_train[indices]
        X_train_aug_kspike = rkspike(X_train_aug)

        # Concatenar os dados aumentados ao dataset original
        X_train = torch.cat([X_train, X_train_aug_gaussian, X_train_aug_gibbs, X_train_aug_kspike], dim=0)
        y_train = torch.cat([y_train, y_train_aug_gaussian, y_train_aug_gibbs, y_train_aug_kspike], dim=0)
        del X_train_aug, X_train_aug_gaussian, X_train_aug_gibbs, X_train_aug_kspike,y_train_aug_gaussian, y_train_aug_gibbs, y_train_aug_kspike
        torch.cuda.empty_cache()
        while True:
            # Automatizo la alocacion inicial de la memoria
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
            # empiezo a llamar la funcion de entrenamiento
            # caso llegue a la memoria maxima de la gpu, reduzco el batch_size o intento cambiar de gpu
            try:
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
        if(metric > best_metric):
            best_metric = metric
            save_model(model_copy, f'./models_class/{name}.pth')
        torch.cuda.empty_cache()
        model_copy = load_model(f'./models_class/{name}.pth', model_copy)
        with torch.no_grad():
            metricas = predict(model_copy, X_val, y_val, batch_size)
        with open("results.txt", "a") as f: #guardo las prediciones y el true label para cada fold
            f.write("name: " + name + f"Fold {fold + 1}""\n")
            f.write("y_test: " + str(y_val) + "\n")
            f.write("y_pred: " + str(metricas["predictions"]) + "\n") # numeros + es para classe 1 y - classe 0
        # despues de cada fold, hago la interpretacion de las imagenes con el mejor modelo
        smothgradcampp(model_copy, X_val, y_val, nameImg=name+f"_fold_{fold+1}", P=metricas["predictions"])
        del X_val, y_val
        torch.cuda.empty_cache()
        print(f"Metricas Validacion Fold {fold + 1}: ", metricas["metrics"])
        fold_accuracies.append(metricas["metrics"]["accuracy"])
        fold_recall.append(metricas["metrics"]["recall"])
        fold_precision.append(metricas["metrics"]["precision"])
        fold_f1.append(metricas["metrics"]["f1_score"])
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
    with open("results.txt", "a") as f: #guarda las metricas de validadcion para cada fold
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

# funcion para hacer prediciones y calcular metricas
def predict(model, X, Y, batch_size=32):
    device = next(model.parameters()).device
    fn_loss = nn.BCEWithLogitsLoss(torch.tensor(274/226).to(device))
    X = torch.Tensor(X).float()
    Y = torch.tensor(Y, dtype=torch.uint8)
    loader = data.DataLoader(data.TensorDataset(X, Y), batch_size=batch_size, shuffle=False)    
    model.eval() # modelo en modo de evaluacion
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = fn_loss(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item() # no hago el backpropagation
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

def smothgradcampp(model, X, Y, nameImg=None, P=None):
    # aplico interpretabilidad a los modelos para saber cual va realmente mejor
    device = next(model.parameters()).device
    X = torch.Tensor(X).float().to(device)
    Y = torch.tensor(Y, dtype=torch.uint8).to(device)
    last_conv_layer = get_last_conv_layer(model)
    model.eval()
    cam_extractor = SmoothGradCAMpp(model, target_layer=last_conv_layer, input_shape=(1, 512, 512))
    indices = np.random.choice(len(X), 2, replace=False)
    for i in indices:
        aux = X[i].unsqueeze(0).to(device) # 1, 1 ,512 , 512
        out = model(aux)  # Forward pass
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        img_rgb = to_pil_image(X[i]).convert('RGB') 
        result = overlay_mask(img_rgb, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        result.save(f'./interpreter/{nameImg}_{i}_class_{Y[i]}_pred_{1 if out[0][0] > 0 else 0}.png')
        torch.cuda.empty_cache()
        del aux, out, activation_map, result
    torch.cuda.empty_cache()



def get_last_conv_layer(model):
    # intenta cojer el ultimo conv2D
    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Conv2d):
            return layer
    return list(model.modules())[-2]