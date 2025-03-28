from models.custom_VGG16 import PyTorchVGG16Logits
from models.custom_VGG19 import PyTorchVGG19Logits
from models.custom_Resnet34 import ResNet34
from models.custom_Resnet50 import ResNet50
from models.convit import VisionTransformer as vt
from auxiliares.fit import train

"""
Funcion que llama los modelos para entrenar con las imagenes rotadas
"""
def pos(X, Y):
    model = PyTorchVGG16Logits()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "vgg16_pos")
    model = PyTorchVGG19Logits()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5, epochs=500, batch_size=64, name ="vgg19_pos")
    model = ResNet34()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5, epochs=500, batch_size=64, name ="resnet_pos")
    model = ResNet50()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name ="resnet50_pos")
    # model = vt()
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "convit_pos")
    # model = fast_vggkan(input_channels = 1,num_classes = 1, vgg_type='VGG16')
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "vgg16kan_pos")
    # model = fast_vggkan(input_channels = 1,num_classes = 1, vgg_type='VGG19')
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "vgg19kan_pos")
    # model = reskagnet50(input_channels=1,num_classes=1)
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "resnet50kan_pos")
    # model = reskagnet101(input_channels=1,num_classes=1)
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=64, name = "resnet101kan_pos")