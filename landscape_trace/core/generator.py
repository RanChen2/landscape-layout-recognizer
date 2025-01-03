import torch
import torch.nn as nn
from torch.nn import init
from chen_tool.networks import UnetGenerator

class Generator:
    def __init__(self, gpu_ids, ngf, init_gain, load_path):
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

        # 初始化生成器网络
        net = UnetGenerator(self.input_nc, self.output_nc, 8, self.ngf, use_dropout=False)

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

        # GPU处理（仅移动到GPU，暂不包装DataParallel）
        if gpu_ids and torch.cuda.is_available():
            net.to(gpu_ids[0])

        self.netG = net

        # 加载预训练模型
        state_dict = torch.load(self.load_path, map_location=str(self.device))

        # 处理模型状态字典
        if hasattr(state_dict, '_metadata'): 
            del state_dict._metadata

        for key in list(state_dict.keys()):
            parts = key.split('.')
            obj = self.netG
            for part in parts[:-1]:
                obj = getattr(obj, part)
            if obj.__class__.__name__.startswith('InstanceNorm'):
                attr = parts[-1]
                if attr in ['running_mean', 'running_var', 'num_batches_tracked'] and getattr(obj, attr,None) is None:
                    state_dict.pop(key)

        # 加载处理后的状态字典到模型中
        self.netG.load_state_dict(state_dict)

        # 在加载完状态字典后再包装DataParallel
        if gpu_ids and torch.cuda.is_available():
            self.netG = torch.nn.DataParallel(self.netG, gpu_ids) 