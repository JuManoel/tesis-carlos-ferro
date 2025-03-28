import pandas as pd
import os
from auxiliares.preProcesamentoImg import *
from imgs.fstd import *
from imgs.fstdr import *
from imgs.pos import *
from imgs.std import *
from imgs.stdr import *
from imgs.normales import *

"""
Script con las responsabilidades de:
1. Limpiar el directorio de trabajo, results.txt, interpreter y los .pth para liberar armazenamiento
2. Cargar las imagenes ya preprocesadas
3. Entrenar los modelos con las imagenes preprocesadas
4. Mostrar los resultados en el archivo results.txt
"""

# cuando inicia un entrenamiento ella limpia el archivo results.txt
path = "/data/home/rtabares/carlos-ferro/"
results_file = os.path.join(path, "results.txt")
with open(results_file, 'w') as file:
    file.write('')
    # borra todas las imagenes de la interpretabilidad
    interpreter_dir = os.path.join(path, "interpreter")
    if os.path.exists(interpreter_dir):
        for filename in os.listdir(interpreter_dir):
            file_path = os.path.join(interpreter_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    # borra todos los modelos ya preentrenados
    models_class = os.path.join(path, "models_class")
    if os.path.exists(models_class):
        for filename in os.listdir(models_class):
            file_path = os.path.join(models_class, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


# valido si el directoria de las imagenes preprocesadas existe
images_dir = os.path.join(path, "preprosImg")
if not os.path.exists(images_dir): raise FileNotFoundError(f"The directory {images_dir} does not exist.")

# las organizo por nombre
files = sorted([
    f for f in os.listdir(images_dir) 
    if os.path.isfile(os.path.join(images_dir, f))
])

# cargo el archivo excel con las clases
excel = pd.read_excel(path+"data_dental/clases.xlsx")
prepros = Preprocess()
# cargo unicamente las fstdr, fstd, std, stdr, ya que las normales y la pos no generan buenos resultados
_,_, X_fstd,Y_fstd,X_std,Y_std, X_fstdr,Y_fstdr,X_stdr,Y_stdr,_,_ = prepros(images_dir, files, excel)

# entreno los modelos
#normales(X,Y)
"""fstdr"""
fstdr(X_fstdr,Y_fstdr)  
"""fstd"""
fstd(X_fstd,Y_fstd)
"""stdr"""
stdr(X_stdr,Y_stdr)
"""std"""
std(X_std,Y_std)
"""pos"""
#pos(X_pos,Y_pos)

# muestro los resultados en el archivo results.txt
with open(results_file, 'r') as file:
    contents = file.read()
    print(contents)