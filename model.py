import torch
import segmentation_models_pytorch as smp

def U_NET(encoder="resnet34", encoder_weights="imagenet", classes=3, activation=None):
    """
    Hàm này tạo ra một mô hình U-Net mạnh mẽ.

    - encoder_name (str): Tên của mạng CNN dùng làm bộ mã hóa (backbone). 
                          "resnet34" là một lựa chọn cân bằng giữa tốc độ và độ chính xác.
    - encoder_weights (str): "imagenet" nghĩa là chúng ta sẽ dùng các trọng số đã được
                             huấn luyện trước trên bộ dữ liệu ImageNet khổng lồ. 
                             Điều này giúp mô hình của bạn học nhanh hơn và tốt hơn rất nhiều.
    - in_channels (int): Số kênh màu của ảnh đầu vào (ảnh màu RGB là 3).
    - classes (int): Số lớp bạn muốn phân loại. Trong trường hợp của bạn là 3 lớp:
                     0: Nền (background)
                     1: Vạch liền (solid)x`
                     2: Vạch đứt (dashed)
    - activation: Hàm kích hoạt ở lớp cuối cùng. Để là None (mặc định) là tốt nhất khi
                  sử dụng hàm mất mát CrossEntropyLoss trong PyTorch.
    """
    model = smp.Unet(
        encoder_name=encoder,        
        encoder_weights=encoder_weights, 
        in_channels=3,                  
        classes=classes,                 
        activation=activation,            
    )
    return model
