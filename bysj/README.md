##基于YOLOv8和CLIP的多维度照片智能分析系统的设计与实现##

## 项目简介

本项目是一个基于深度学习与传统图像处理相结合的多维度照片智能分析系统。系统支持街拍、风光、建筑等多题材照片的自动分析与打分，涵盖曝光、对焦、噪点、构图、色调、光影、内容表达等7大维度，并给出优化建议。项目采用 Streamlit 搭建本地 Web 界面，支持图片上传、主题输入、实时分析和结果展示。

---

## 主要功能

- **曝光分析**：自动检测照片曝光情况，判断过曝、欠曝或正常。
- **对焦分析**：评估照片清晰度，检测是否模糊。
- **噪点分析**：估算照片噪点水平，给出降噪建议。
- **构图分析**：基于 YOLOv8 目标检测，结合三分法、黄金分割点、主体边界距离等美学规则，智能分析构图质量。
- **色调分析**：检测主色调占比，判断色调统一性。
- **光影分析**：评估照片对比度，分析光影层次。
- **内容表达分析**：集成 OpenAI CLIP 模型，支持输入主题词，自动计算图片与主题的相关性分数。
- **智能建议**：根据各维度得分，自动生成优化建议。
- **可视化界面**：Streamlit Web 界面，支持图片上传、主题输入、结果展示与可视化。

---

## 环境依赖

- Python 3.8~3.10（推荐3.9）
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [Numpy](https://numpy.org/)
- [Streamlit](https://streamlit.io/)
- [ultralytics](https://github.com/ultralytics/ultralytics)（YOLOv8）
- [torch](https://pytorch.org/)
- [clip](https://github.com/openai/CLIP)（OpenAI官方库）
- 其他依赖：`Pillow`, `tqdm`, `ftfy`, `regex`

建议使用如下 `requirements.txt`：

```txt
opencv-python
numpy
streamlit
ultralytics
torch
git+https://github.com/openai/CLIP.git
Pillow
tqdm
ftfy
regex
```

---

## 安装与运行

1. **克隆项目并进入目录**
   ```bash
   git clone <你的项目地址>
   cd bysj
   ```

2. **安装依赖**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **运行 Streamlit 服务**
   ```bash
   streamlit run realtime_photo_assistant.py
   ```

4. **在浏览器访问**
   - 打开终端提示的 `http://localhost:8501`（或其他端口）即可使用系统。

---

## 使用说明

1. 打开 Web 页面后，点击“上传照片”选择你要分析的图片（支持 jpg、jpeg、png）。
2. 在“主题词/短语”输入框中填写照片主题（如“城市夜景”、“自然风光”、“现代建筑”等）。
3. 系统会自动分析照片的7大维度，给出每项得分、综合得分和优化建议，并可视化构图分析结果。
4. 根据建议优化你的拍摄或后期处理。

---

## 主要技术点

- **OpenCV**：实现曝光、对焦、噪点、色调、光影等基础图像分析与可视化。
- **YOLOv8（ultralytics）**：目标检测，智能分析构图（支持三分法、黄金分割点、边界惩罚、多主体分布等）。
- **OpenAI CLIP**：内容表达分析，计算图片与主题的相关性分数。
- **Streamlit**：搭建本地 Web 界面，支持交互式操作和结果展示。

---

## 常见问题

- **依赖安装失败**：请确保 Python 版本和 pip3 可用，必要时升级 pip。
- **模型权重下载慢**：首次运行 YOLOv8/CLIP 时需联网自动下载权重，耐心等待即可。
- **端口被占用**：Streamlit 会自动切换端口，终端会有提示。

---

## 项目亮点与扩展方向

- 多维度智能分析，兼顾传统美学与AI能力。
- 支持多题材、多场景，适用面广。
- 可扩展性强，后续可集成更多AI模型或功能（如批量分析、历史记录、自动主题生成等）。

---

## 联系与反馈

如有问题或建议，欢迎联系作者或在项目仓库提交 issue。 