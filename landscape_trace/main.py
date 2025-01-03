import time
import os
from PIL import Image, ImageOps
from chen_tool.watermark import get_start_banner, get_end_banner
from core.generator import Generator
from core.image_process import resize_image, pre_process
from core.utils import tensor2im, im2tensor
from configs.config_loader import config

def main():
    print(get_start_banner())

    # 验证模型文件路径
    if not config.validate_paths():
        print("错误：无法找到所需的模型文件！")
        return

    total_start_time = time.time()
    step_start_time = time.time()
    
    print('\n' + '='*50)
    print('【步骤1：模型加载】')
    models = {}
    model_paths = config.get_all_model_paths()
    model_params = config.get_model_params()
    
    for path in model_paths:
        models[path] = Generator(
            gpu_ids=model_params['gpu_ids'],
            ngf=model_params['ngf'],
            init_gain=model_params['init_gain'],
            load_path=path
        )
        print(f'- {path} 模型加载完成')
    step_time = time.time() - step_start_time
    print(f'模型加载总耗时: {step_time:.2f}秒')
    print('='*50 + '\n')

    # 打开输入图
    step_start_time = time.time()
    print('【步骤2：图像预处理】')
    input_image_path = '000218.png'
    output_image_path = 'test.png'

    input_image = Image.open(input_image_path)
    image_params = config.get_image_params()
    input_image = resize_image(input_image, max_size=image_params['max_size'])
    print(f'- 图像缩放完成，当前尺寸: {input_image.size}')
    step_time = time.time() - step_start_time
    print(f'图像预处理耗时: {step_time:.2f}秒')
    print('='*50 + '\n')

    print('【步骤3：外环境处理】')
    step_start_time = time.time()
    # 生成mask
    model = models[config.get_model_path('mask')]
    input_tensor = im2tensor(input_image)
    output_tensor = model.netG(input_tensor)
    external_mask = tensor2im(output_tensor).resize(input_image.size)
    print('- 外环境识别完成')

    # 优化mask
    external_mask_optimize = pre_process('mask', external_mask, True, input_image.size)
    print('- 外轮廓优化完成')

    # 合并mask和输入图像
    black_image = Image.new("RGB", input_image.size, (0, 0, 0))
    mask = external_mask_optimize.convert('L')
    cover_external_mask = Image.composite(input_image, black_image, mask).resize(input_image.size)
    print('- 外环境去除完成')
    step_time = time.time() - step_start_time
    print(f'外环境处理耗时: {step_time:.2f}秒')
    print('='*50 + '\n')

    print('【步骤4：分层识别】')
    step_start_time = time.time()
    landuse_dict = {'mask': external_mask_optimize}
    process_order = config.get_process_order()
    
    for name in process_order:
        model = models[config.get_model_path(name.lower())]
        input_tensor = im2tensor(cover_external_mask)
        output_tensor = model.netG(input_tensor)
        mask = tensor2im(output_tensor).resize(input_image.size)
        mask = pre_process(name, mask, False, input_image.size)
        landuse_dict[name] = mask
        print(f'- {name} 识别完成')
    step_time = time.time() - step_start_time
    print(f'分层识别总耗时: {step_time:.2f}秒')
    print('='*50 + '\n')

    print('【步骤5：图层赋色】')
    step_start_time = time.time()
    canvas = Image.new('RGBA', input_image.size, color=(0, 0, 0, 255))
    color_map = config.get_color_map()
    
    for name in landuse_dict:
        landuse = landuse_dict[name]
        landuse_gray = landuse.convert("L")
        color = tuple(color_map[name])
        colored_landuse = ImageOps.colorize(landuse_gray, black='black', white=color).convert("RGBA")
        mask = landuse_gray.point(lambda p: 255 if p > 128 else 0).convert("L")
        canvas.paste(colored_landuse, (0, 0), mask)
        print(f'- {name} 赋色完成')
    step_time = time.time() - step_start_time
    print(f'图层赋色总耗时: {step_time:.2f}秒')
    print('='*50 + '\n')

    print('【步骤6：保存结果】')
    step_start_time = time.time()
    canvas.save(output_image_path.replace('.jpg','.png'), 'PNG')
    step_time = time.time() - step_start_time
    print(f'- 结果已保存至: {output_image_path}')
    print(f'保存耗时: {step_time:.2f}秒')
    print('='*50)

    total_time = time.time() - total_start_time
    print(f'\n总处理时间: {total_time:.2f}秒')
    print('='*50)
    
    print(get_end_banner())

if __name__ == '__main__':
    main() 