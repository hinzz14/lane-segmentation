import torch
import segmentation_models_pytorch as smp

def U_NET(encoder="resnet34", encoder_weights="imagenet", classes=3, activation=None):
    model = smp.Unet(
        encoder_name=encoder,        
        encoder_weights=encoder_weights, 
        in_channels=3,                  
        classes=classes,                 
        activation=activation,            
    )
    return model
