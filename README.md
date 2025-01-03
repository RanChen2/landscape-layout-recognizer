# 风景园林平面图描图机 v1.0

这是一个从平面图生成设计布局数据的工具。本项目是论文《风景园林平面图生成设计数据集增强方法研究》的配套代码实现。

## 示例结果

您可以在以下链接查看一些处理结果示例：
https://cloud.tsinghua.edu.cn/d/e7fd019a7b2444e0add0/

## 开发团队

@地球研究所ppt @罗晓敏 @陈博士开发

联系我们：
- 微信咨询：15690576620（罗晓敏）/7053677787（陈博士）
- 小红书/B站：地球研究所PPT
- 微信公众号：地球研究社PPT

## 引用

如果您在研究中使用了本项目，请引用以下论文：
```
@article{ ZGYL202409005,
author = { 陈然 and  罗晓敏 and  凌霄 and  赵晶 },
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

## 功能特点

- 自动识别并分离园林平面图要素
- 智能处理图层合成与优化
- 专业数据集生成与标注
- 支持多种园林元素的识别，包括：
  - 外部环境
  - 绿地（LD）
  - 水体（ST）
  - 铺装+道路（PZ+DL）
  - 铺装（PZ）
  - 构筑物（GZW）
  - 植物（ZW）

## 安装说明

1. 克隆项目到本地：
```bash
git clone [repository_url]
cd landscape_trace
```

2. 创建并激活虚拟环境（推荐）：
```bash
conda create -n landscape python=3.8
conda activate landscape
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载预训练模型：
请从以下链接下载预训练模型文件（.pth）并放置在 `weights` 目录下：
【待上传】

## 使用方法

1. 准备输入图像：
   - 支持常见图像格式（PNG、JPG等）
   - 建议图像分辨率不超过1024像素

2. 运行程序：
```bash
python main.py
```

3. 配置说明：
   - 在 `configs/config.json` 中可以调整各项参数
   - 支持自定义处理顺序和颜色映射



## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。 