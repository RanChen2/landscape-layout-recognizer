import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def tensor2im(input_tensor, imtype=np.uint8):
    """将tensor转换为图像"""
    image_tensor = input_tensor.data
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    output_image = image_numpy.astype(imtype)
    output_image = Image.fromarray(np.uint8(output_image))
    return output_image

def im2tensor(input_image):
    """将图像转换为tensor"""
    input_image = input_image.convert('RGB')
    transform_list = [
        transforms.Resize([512,512], interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)
    output_tensor = transform(input_image)
    output_tensor = torch.unsqueeze(output_tensor, 0)
    return output_tensor 