# 这是一个示例 Python 脚本。
import torch
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

class generator():
    def __init__(self, gpu_ids,ngf,init_gain,load_path):
        self.gpu_ids = gpu_ids
        self.device = torch.device(f'cuda:{gpu_ids[0]}') if gpu_ids else torch.device('cpu')
        self.load_size = 512
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = ngf
        self.init_gain = init_gain
        self.load_path = load_path
        self.real_A = None
        self.fake_B = None
        # torch.backends.cudnn.benchmark = True # 加速卷积，在图像一样大的时候可以加速

        # ############### 初始化生成器 G
        # 定义网络
        net = UnetGenerator(self.input_nc, self.output_nc, 8, self.ngf, use_dropout=False)

        # GPU处理
        if gpu_ids and torch.cuda.is_available():
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # Wrap the network into DataParallel if multiple GPUs are used

        # 初始化权重
        for m in net.modules():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
                init.normal_(m.weight.data, 0.0, self.init_gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif 'BatchNorm2d' in classname:
                init.normal_(m.weight.data, 1.0, self.init_gain)
                init.constant_(m.bias.data, 0.0)

        self.netG = net

        # 3 加载pth
        state_dict = torch.load(self.load_path, map_location=str(self.device))

        # ######## 超长防呆设计
        if hasattr(state_dict, '_metadata'): del state_dict._metadata
        for key in list(state_dict.keys()): # 对于使用InstanceNorm的网络层，处理特定的状态键
            parts = key.split('.')
            obj = self.netG
            for part in parts[:-1]:  # 遍历到倒数第二个元素，定位到具体的层
                obj = getattr(obj, part)
            # 检查层的类型，是否是InstanceNorm
            if obj.__class__.__name__.startswith('InstanceNorm'):
                attr = parts[-1]
                # 对于InstanceNorm不应该跟踪的属性，如运行均值和方差，进行特殊处理
                if attr in ['running_mean', 'running_var', 'num_batches_tracked'] and getattr(obj, attr,None) is None:
                    state_dict.pop(key)  # 从状态字典中移除这些键

        # 加载处理后的状态字典到模型中
        self.netG.load_state_dict(state_dict)


class UnetGenerator(nn.Module):
    # 构造函数，初始化生成器
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=False):
        super(UnetGenerator, self).__init__()  # 调用父类构造函数
        # 构建U-Net结构，从内层到外层逐层嵌套
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # 最内层
        for i in range(num_downs - 5):  # 添加中间层
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # 逐步减少滤波器数量
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # 最外层

    # 前向传播函数
    def forward(self, input):
        return self.model(input)  # 通过模型处理输入并返回结果
class UnetSkipConnectionBlock(nn.Module):
    # 构造函数，定义块的配置
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()  # 调用父类构造函数
        self.outermost = outermost  # 是否为最外层
        # 判断是否使用偏置
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc  # 如果没有指定输入通道数，则设为外层通道数
        # 定义向下的卷积操作
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        # 定义向上的卷积操作
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # 根据层的位置（最外层、最内层或中间层）配置不同的模块组合
        if outermost:  # 最外层，输出层使用Tanh激活
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:  # 最内层，没有跳跃连接
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:  # 中间层，包含跳跃连接
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:  # 如果使用dropout
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)  # 将所有层组合成一个序列模型

    # 前向传播函数
    def forward(self, x):
        if self.outermost:  # 如果是最外层，直接返回结果
            return self.model(x)
        else:  # 否则，加上输入（实现跳跃连接）
            return torch.cat([x, self.model(x)], 1)  # 在通道维度上连接输入和输出，实现U-Net的典型特性

# def tensor2im(input_tensor, imtype=np.uint8):
#     image_tensor = input_tensor.data
#     image_numpy = image_tensor[0].cpu().float().numpy()  # 变成numpy
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # 后处理：变换和缩放
#     output_image = image_numpy.astype(imtype)
#     output_image = Image.fromarray(np.uint8(output_image))
#     return output_image # 返回numpy
def tensor2im(input_image, imtype=np.uint8):
    image_tensor = input_image.data
    image_numpy = image_tensor[0].cpu().float().numpy()  # 变成numpy
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # 后处理：变换和缩放
    return image_numpy.astype(imtype) # 返回numpy

def im2tensor(input_image_path):
    input_image = Image.open(input_image_path).convert('RGB')
    transform_list = [
        transforms.Resize([512,512], InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)

    output_tensor = transform(input_image)
    output_tensor = torch.unsqueeze(output_tensor, 0)

    return output_tensor