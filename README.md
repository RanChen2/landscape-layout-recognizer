# 风景园林平面图描图机 v1.0
# Landscape Architecture Plan Tracer v1.0

- 这是一个从平面图生成设计布局数据的工具。
- This is a tool for generating design layout data from landscape architecture plans.

- 输入一张风景园林平面图，输出一张设计布局图。
- Input a landscape architecture plan and output a design layout.

- 用于生产设计平面图合成数据，支撑风景园林平面图生成算法训练。
- Used to produce synthetic data for training landscape architecture plan generation algorithms.

- 本项目是论文《风景园林平面图生成设计数据集增强方法研究》的配套代码实现。
- This project is the implementation of the paper "Research on Enhancement Methods for Generating Design Datasets for Landscape Architecture Plans".

## 示例结果 | Example Results

您可以在以下链接查看一些处理结果示例：

You can view some processing results at the following link:

[生成结果示例链接1000张 | 1000 Generated Examples](https://cloud.tsinghua.edu.cn/d/e7fd019a7b2444e0add0/)

## 开发团队 | Development Team

@地球研究所PPT @小敏姐姐 @陈博士 开发

Developed via @Xiaomin @Dr. Chen

联系我们：
- 微信咨询：15690576620（小敏姐姐）/7053677787（陈博士）
- 小红书/B站：地球研究所PPT
- 微信公众号：地球研究社PPT

Contact Us:
- WeChat: 15690576620 (Xiaomin) / 7053677787 (Dr. Chen)
- Xiaohongshu/Bilibili: Earth Research Institute PPT
- WeChat Official Account: Earth Research Society PPT

## 引用 | Citation

如果您在研究中使用了本项目，请引用以下论文：

If you use this project in your research, please cite the following paper:

中文知网引用 | Chinese Citation:
```bibtex
@article{ZGYL202409005,
    author = {陈然 and 罗晓敏 and 凌霄 and 赵晶},
    title = {风景园林平面图生成设计数据集增强方法研究},
    journal = {中国园林},
    volume = {40},
    number = {09},
    pages = {36-42},
    year = {2024},
    issn = {1000-6664},
    doi = {10.19775/j.cla.2024.09.0036}
}
```

English Citation:
```bibtex
@article{chen2024enhancement,
    title={Research on Enhancement Methods for Generating Design Datasets for Landscape Architecture Plans},
    author={Chen, Ran and Luo, Xiaomin and Ling, Xiao and Zhao, Jing},
    journal={Chinese Landscape Architecture},
    volume={40},
    number={9},
    pages={36--42},
    year={2024},
    doi={10.19775/j.cla.2024.09.0036},
    note={In Chinese with English abstract}
}
```

## 功能特点 | Features

- 自动识别并分离园林平面图要素
- Automatically identify and separate landscape plan elements

- 智能处理图层合成与优化
- Intelligent layer composition and optimization

- 专业数据集生成与标注
- Professional dataset generation and annotation

支持多种园林元素的识别，包括：

Supports recognition of various landscape elements, including:
  - 外部环境 | External Environment
  - 绿地 | Green Space (LD)
  - 水体 | Water Body (ST)
  - 铺装+道路 | Paving + Road (PZ+DL)
  - 铺装 | Paving (PZ)
  - 构筑物 | Structures (GZW)
  - 植物 | Plants (ZW)

## 安装说明 | Installation

1. 克隆项目到本地 | Clone the repository:
```bash
git clone [repository_url]
cd landscape_trace
```

2. 创建并激活虚拟环境（推荐）| Create and activate virtual environment (recommended):
```bash
conda create -n landscape python=3.8
conda activate landscape
```

3. 安装依赖 | Install dependencies:
```bash
pip install -r requirements.txt
```

4. 下载预训练模型 | Download pre-trained models:

    请从以下链接下载预训练模型文件（.pth）并放置在 `weights` 目录下：

    Please download the pre-trained model files (.pth) from the following link and place them in the `weights` directory:

    [模型权重链接 | Model Weights Link](https://cloud.tsinghua.edu.cn/d/7b2c567ee9a24f08abe1/)

## 使用方法 | Usage

1. 准备输入图像 | Prepare input image:
   - 支持常见图像格式（PNG、JPG等）| Supports common image formats (PNG, JPG, etc.)
   - 建议图像分辨率不超过1024像素 | Recommended image resolution no more than 1024 pixels

2. 运行程序 | Run the program:
```bash
python main.py
```

3. 配置说明 | Configuration:
   - 在 `configs/config.json` 中可以调整各项参数 | Parameters can be adjusted in `configs/config.json`
   - 支持自定义处理顺序和颜色映射 | Supports custom processing order and color mapping

## 许可证 | License

本项目采用 MIT 许可证。详见 LICENSE 文件。
This project is licensed under the MIT License. See the LICENSE file for details. 