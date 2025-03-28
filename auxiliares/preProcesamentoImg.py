import numpy as np
import os
import glob
import cv2

# classe que sirve para cargar todas las imagenes y las guarda en un array
class Preprocess():
    def __init__(self):
        pass

    def __call__(self, path, files, excel):
        X = []
        Y = []
        X_fstdr = []
        Y_fstdr = []
        X_fstd = []
        Y_fstd = []
        X_stdr = []
        Y_stdr = []
        X_std = []
        Y_std = []
        X_pos = []
        Y_pos = []
        print(path)
        image_paths = sorted(glob.glob(os.path.join(path, "*.jpg"))) #gracias a ese comando puedo ahorar mucho tiempo
        print(len(image_paths))
        for i in image_paths:
            name_img = i.split("/")[-1]
            paciente_id = str(name_img[0:6])
            aux = excel.loc[excel['Paciente'] == paciente_id, "Extraccion / no extraccion"]
            if aux.empty:
                continue
            value = 0 if aux.values[0] == 'No extracción' else 1
            if "fstdr" in name_img:
                X_fstdr.append(self.preProcessoImg(i))
                Y_fstdr.append(value)
            elif "fstd" in name_img:
                X_fstd.append(self.preProcessoImg(i))
                Y_fstd.append(value)
            elif "stdr" in name_img:
                X_stdr.append(self.preProcessoImg(i))
                Y_stdr.append(value)
            elif "std" in name_img:
                X_std.append(self.preProcessoImg(i))
                Y_std.append(value)
            elif "pos" in name_img:
                X_pos.append(self.preProcessoImg(i))
                Y_pos.append(value)
            else:
                X.append(self.preProcessoImg(i))
                Y.append(value)
        X = np.array(X,dtype = np.float32)
        Y = np.array(Y, dtype = np.uint8)
        X_fstdr = np.array(X_fstdr,dtype = np.float32)
        Y_fstdr = np.array(Y_fstdr, dtype = np.uint8)
        X_fstd = np.array(X_fstd,dtype = np.float32)
        Y_fstd = np.array(Y_fstd, dtype = np.uint8)
        X_stdr = np.array(X_stdr,dtype = np.float32)
        Y_stdr = np.array(Y_stdr, dtype = np.uint8)
        X_std = np.array(X_std,dtype = np.float32)
        Y_std = np.array(Y_std, dtype = np.uint8)
        X_pos = np.array(X_pos,dtype = np.float32)
        Y_pos = np.array(Y_pos, dtype = np.uint8)
        return X,Y, X_fstdr, Y_fstdr, X_fstd, Y_fstd, X_stdr, Y_stdr, X_std, Y_std, X_pos, Y_pos

    def preProcessoImg(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img/255.0
        img = np.expand_dims(img, axis=0)
        return img