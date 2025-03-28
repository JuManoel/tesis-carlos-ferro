from models.custom_VGG16 import PyTorchVGG16Logits
from models.custom_VGG19 import PyTorchVGG19Logits
from models.custom_Resnet34 import ResNet34
from models.custom_Resnet50 import ResNet50
from models.convit import VisionTransformer as vt
from auxiliares.fit import train
"""
Funcion que llama los modelos para entrenar con las imagenes normales
"""
def normales(X,Y):
    model = PyTorchVGG16Logits()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "vgg16")
    model = PyTorchVGG19Logits()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5, epochs=500, batch_size=64, name ="vgg19")
    model = ResNet34()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5, epochs=500, batch_size=64, name ="resnet")
    model = ResNet50()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name ="resnet50")
    # model = vt()
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "convit")
    # model = fast_vggkan(input_channels = 1,num_classes = 1, vgg_type='VGG16')
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "vgg16kan")
    # model = fast_vggkan(input_channels = 1,num_classes = 1, vgg_type='VGG19')
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "vgg19kan")
    # model = reskagnet50(input_channels=1,num_classes=1)
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "resnet50kan")
    # model = reskagnet101(input_channels=1,num_classes=1)
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "resnet101kan")
    pass