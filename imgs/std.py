from models.custom_VGG16 import PyTorchVGG16Logits
from models.custom_VGG19 import PyTorchVGG19Logits
from models.custom_Resnet34 import ResNet34
from models.custom_Resnet50 import ResNet50
from models.denseNet import densenet121, densenet169
from models.inception import inception_v3
from models.inception_Resnet import InceptionResnetV2
from models.efficientnet.efficientnet import EfficientNet
from auxiliares.fit import train

"""
Funcion que llama los modelos para entrenar con las imagenes std
"""
def std(X, Y):
    model = PyTorchVGG16Logits()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=128, name = "vgg16_std")
    model = PyTorchVGG19Logits()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5, epochs=500, batch_size=128, name ="vgg19_std")
    model = ResNet34()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5, epochs=500, batch_size=128, name ="resnet_std")
    model = ResNet50()
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=128, name ="resnet50_std")
    model = densenet121(dropout_rate=0.5,num_classes=1)
    print(model)
    model = train(model, X, Y, "Lion", num_splits=5,epochs=500, batch_size=128, name ="densenet121_std")
    model = densenet169(dropout_rate=0.5,num_classes=1)
    print(model)
    model = train(model, X, Y, "Lion", num_splits=5,epochs=500, batch_size=128, name ="densenet169_std")
    model = inception_v3(num_classes=1)
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=128, name ="inception_v3_std")
    model = InceptionResnetV2(feature_list_size=1)
    print(model)
    model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=128, name ="inception_resnet_v2_std")
    model = EfficientNet.from_name(model_name='efficientnet-b4',image_size=512-128,num_classes=1)
    print(model)
    model = train(model, X, Y, "Lion", num_splits=5,epochs=500, batch_size=128, name ="efficientnet-b4_std")
    model = EfficientNet.from_name('efficientnet-b7',image_size=512-128,num_classes=1)
    print(model)
    model = train(model, X, Y, "Lion", num_splits=5,epochs=500, batch_size=128, name ="efficientnet-b7_std")
    # model = efficientCapsNet()
    # print(model)
    # model = train(model, X, Y, "AdamW", num_splits=5,epochs=500, batch_size=128, name ="efficientCapsNet_std")