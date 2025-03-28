import torch
import os
import pandas as pd
from .model.meta_class import MetaClassifier
from models.custom_Resnet34 import ResNet34
from models.custom_Resnet50 import ResNet50
from models.custom_VGG16 import PyTorchVGG16Logits
from models.inception import inception_v3
from auxiliares.preProcesamentoImg import Preprocess
from .fit import train

def load_model(PATH, model):
    import io
    with open(PATH, 'rb') as f:
        buffer = io.BytesIO(f.read())
    checkpoint = torch.load(buffer, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main():
    """
    script principal que llama a las funciones de entrenamiento de los modelos,
    carga las imagenes y sus labels
    """

    # cuando inicia un entrenamiento ella limpia el archivo results.txt
    path = "/data/home/rtabares/carlos-ferro/"
    results_file = os.path.join(path, "results_meta.txt")
    # Clear the contents of the file
    with open(results_file, 'w') as file:
        file.write('')
    images_dir = os.path.join(path, "preprosImg")
    """Check if the images directory exists
    """
    if not os.path.exists(images_dir): raise FileNotFoundError(f"The directory {images_dir} does not exist.")
    """List and sort files in the images directory
    """
    files = sorted([
        f for f in os.listdir(images_dir) 
        if os.path.isfile(os.path.join(images_dir, f))
    ])

    """Output the sorted list of image files
    """
    excel = pd.read_excel(path+"data_dental/clases.xlsx")
    prepros = Preprocess()
    _,_, X_fstd,Y_fstd,X_std,Y_std, X_fstdr,Y_fstdr,X_stdr,Y_stdr,_,_ = prepros(images_dir, files, excel)

    resnet = ResNet34()
    resnet50 = ResNet50()
    vgg16 = PyTorchVGG16Logits()
    inception = inception_v3(num_classes=1)

    meta_fstdr = MetaClassifier([load_model(f"{path}models_class/resnet_fstdr.pth",resnet), 
                                load_model(f"{path}models_class/resnet50_fstdr.pth",resnet50), 
                                load_model(f"{path}models_class/vgg16_fstdr.pth",vgg16), 
                                load_model(f"{path}models_class/inception_v3_fstdr.pth",inception)], num_classes=1)
    train(meta_fstdr, X_fstdr, Y_fstdr, "AdamW", num_splits=5, epochs=20, batch_size=128, name="meta_fstdr")

    meta_fstd = MetaClassifier([load_model(f"{path}models_class/resnet_fstd.pth",resnet), 
                                load_model(f"{path}models_class/resnet50_fstd.pth",resnet50), 
                                load_model(f"{path}models_class/vgg16_fstd.pth",vgg16), 
                                load_model(f"{path}models_class/inception_v3_fstd.pth",inception)], num_classes=1)
    train(meta_fstd, X_fstd, Y_fstd, "AdamW", num_splits=5, epochs=20, batch_size=128, name="meta_fstd")

    meta_stdr = MetaClassifier([load_model(f"{path}models_class/resnet_stdr.pth",resnet), 
                                load_model(f"{path}models_class/resnet50_stdr.pth",resnet50), 
                                load_model(f"{path}models_class/vgg16_stdr.pth",vgg16), 
                                load_model(f"{path}models_class/inception_v3_stdr.pth",inception)], num_classes=1)
    train(meta_stdr, X_stdr, Y_stdr, "AdamW", num_splits=5, epochs=20, batch_size=128, name="meta_stdr")

    meta_std = MetaClassifier([load_model(f"{path}models_class/resnet_std.pth",resnet), 
                                load_model(f"{path}models_class/resnet50_std.pth",resnet50), 
                                load_model(f"{path}models_class/vgg16_std.pth",vgg16), 
                                load_model(f"{path}models_class/inception_v3_std.pth",inception)], num_classes=1)
    train(meta_std, X_std, Y_std, "AdamW", num_splits=5, epochs=20, batch_size=128, name="meta_std")