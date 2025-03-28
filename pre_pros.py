import cv2
import numpy as np
import os
import glob
from datetime import datetime
# libreria que seirva para programacion concurrente
from concurrent.futures import ThreadPoolExecutor
# libreria de monai que sirve para problemas de imagenes meticas (preprocesar imagenes)
from monai.transforms import Resize 

# Clase que coje las imagenes y las corta en 128x128 de derecha a izquierda y de arriba a abajo
# y las guarda en el directorio preprosImg
class Preprocess():
    def __init__(self):
        pass

    def __call__(self, path):
        print(path)
        image_paths = sorted(glob.glob(os.path.join(path, "*.jpg")))
        print(len(image_paths))
        os.makedirs("./preprosImg", exist_ok=True)
        start = datetime.now().strftime("%H:%M:%S")
        print(f"Start time: {start}")

        with ThreadPoolExecutor() as executor:
            executor.map(self.process_and_save_image, image_paths)

        end = datetime.now().strftime("%H:%M:%S")
        print(f"End time: {end}")
        print("Todas imagenes guardadas con exito")

    def process_and_save_image(self, img_path):
        name_img = img_path.split("/")[-1]
        img = self.preProcessoImg(img_path)
        output_path = os.path.join("./preprosImg", name_img)
        cv2.imwrite(output_path, img)

    def preProcessoImg(self, img_path):
        resizef = Resize((512, 512))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=0)
        img = resizef(img)
        img = np.squeeze(img, axis=0)
        img = img[128:512, 128:512]
        return img

pre = Preprocess()
path = "/data/home/rtabares/carlos-ferro/"
images_dir = os.path.join(path, "data_dental", "full_dataset")
pre(images_dir)