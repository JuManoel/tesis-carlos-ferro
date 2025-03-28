import torch

"""
funcion que guarda el modelo, utilizada principalmente para earling stopping
"""
def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)

"""
funcion que carga el modelo. Utilizada para cojer el mejor modelo del entrenamiento
"""
def load_model(PATH, model):
    import io
    with open(PATH, 'rb') as f:
        buffer = io.BytesIO(f.read())
    checkpoint = torch.load(buffer)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model