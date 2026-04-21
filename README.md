# 图片文本搜索工具 v0.0.1

一个用于识别图片中文字并搜索特定文本的工具，支持中文、数字、英文等文本。

## 功能特点
- 支持识别图片中的文本（中文、英文、数字）
- 搜索包含指定文本的图片
- 自动复制匹配的图片到桌面文件夹
- 简单直观的GUI界面
- 支持Windows 10系统
- 双OCR引擎：Tesseract(兼容) / PaddleOCR-GPU(高性能)
- 三种识别模式：极速 / 平衡 / 精准
- GPU/核显加速预处理（实验性）
- 智能缓存索引系统
- 并行多线程OCR处理

## 安装说明

### 1. 安装Python
确保你的系统已安装Python 3.6或更高版本。

### 2. 安装依赖库
```bash
pip install pytesseract pillow
```

可选GPU加速依赖：
```bash
pip install opencv-python numpy
```

可选PaddleOCR GPU依赖：
```bash
pip install paddlepaddle-gpu paddleocr
```

### 3. 安装Tesseract OCR引擎
1. 下载：https://github.com/UB-Mannheim/tesseract/wiki
2. 安装中文语言包：https://github.com/tesseract-ocr/tessdata
3. 将 `chi_sim.traineddata` 复制到 tessdata 目录

## 使用方法
1. 运行 `image_text_search.py`
2. 输入要查找的文本
3. 选择图片文件夹
4. 选择OCR引擎和识别模式
5. 点击"开始搜索"

## 支持的图片格式
JPG/JPEG, PNG, BMP, TIFF, GIF

## 系统要求
- Windows 10+
- Python 3.6+
- 至少2GB内存