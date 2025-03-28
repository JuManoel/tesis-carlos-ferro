# Predicción de Extracción Dental Basada en Radiografías

Este proyecto tiene como objetivo predecir la extracción o no extracción de dientes utilizando imágenes de radiografías dentales. Se implementan modelos de aprendizaje automático y aprendizaje profundo para analizar las imágenes y generar predicciones precisas.

## Estructura del Proyecto

El proyecto está organizado en las siguientes carpetas principales:

### 1. `auxiliares`
Contiene scripts auxiliares para el preprocesamiento y manejo de modelos:
- `EarlyStopping.py`: Implementación de early stopping para el entrenamiento.
- `fit.py`: Funciones para entrenar modelos.
- `metricas.py`: Cálculo de métricas de evaluación.
- `preProcesamentoImg.py`: Preprocesamiento de imágenes.
- `save_load_model.py`: Guardado y carga de modelos.

### 2. `meta_class`
Incluye scripts relacionados con la clasificación:
- `fit.py`: Entrenamiento de modelos de clasificación.
- `main.py`: Script principal para la ejecución del flujo completo.
- `model/meta_class.py`: Implementación de la clase principal del modelo.

### 3. `models`
Contiene arquitecturas de modelos personalizados:
- `convit.py`
- `custom_Resnet34.py`
- `custom_Resnet50.py`
- `custom_VGG16.py`
- `custom_VGG19.py`
- `denseNet.py`
- `efficientCapsNet.py`
- Otros modelos adicionales.

### 4. `imgs`
Scripts para el manejo de diferentes tipos de imágenes:
- `fstd.py`
- `fstdr.py`
- `normales.py`
- `pos.py`
- `std.py`
- `stdr.py`

## Requisitos

Para ejecutar este proyecto, asegúrate de instalar las dependencias listadas en el archivo `requirements.txt`. Puedes instalarlas con el siguiente comando:

```bash
pip install -r requirements.txt
```

## Flujo del Proyecto

1. **Preprocesamiento de Imágenes**:  
    Las imágenes de radiografías se preprocesan utilizando el script `preProcesamentoImg.py`. Este paso incluye el redimensionamiento y recorte de las imágenes.

2. **Entrenamiento de Modelos**:  
    Los modelos se entrenan utilizando las imágenes preprocesadas. El script `main.py` organiza el flujo completo, desde la limpieza del entorno hasta la generación de resultados.

3. **Evaluación**:  
    Los resultados del entrenamiento se almacenan en el archivo `results.txt` y se pueden analizar para evaluar el rendimiento de los modelos.

## Nota Importante

Si necesitas las imágenes de radiografías utilizadas en este proyecto, por favor comunícate con nosotros para obtener acceso. Estas imágenes son esenciales para replicar los resultados y realizar pruebas adicionales.

## Contacto

Para cualquier consulta o solicitud de acceso a los datos, no dudes en contactarnos.  