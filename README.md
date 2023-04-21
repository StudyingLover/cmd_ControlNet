# ControlNet代码改造计划
虽然现在webui已经支持了ControlNet，但是如果我们需要单独抽出来ControlNet做一些项目就需要对ControlNet进行改造。同时我也想加入一些开源的工具让ControlNet更加有趣，例如[clip_interrogator](https://github.com/pharmapsychotic/clip-interrogator).

关于什么是Canny算子，Hough算子，可以看北邮鲁鹏老师的课程[计算机视觉（本科）北京邮电大学 鲁鹏](https://www.bilibili.com/video/BV1nz4y197Qv/?spm_id_from=333.999.0.0&vd_source=e8f062c423dc7ce759a573dd732735a0)

如果你想在webui使用ControlNet，可以看我之前的[文章](https://studyinglover.com/2023/03/20/%E9%80%9A%E8%BF%87colab%E4%BD%93%E9%AA%8CControlNet/)  ，或者直接查看[webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  

项目开源在[GitHub](https://github.com/StudyingLover/cmd_ControlNet)
我的博客[https://studyinglover.com](https://studyinglover.com/2023/04/21/ControlNet%E4%BB%A3%E7%A0%81%E6%94%B9%E9%80%A0/)

ControlNet项目主页
[github](https://github.com/lllyasviel/ControlNet)
[huggingface](https://huggingface.co/lllyasviel/ControlNet)

推荐一个教程[Stable Diffusion 公测了，开个问题来晒图吧? - 代码狂魔的回答 - 知乎](
https://www.zhihu.com/question/550101073/answer/2931261853)

## TODO
- [x] CLIP_interrogator
- [ ] ControlNet关键函数提取 
- [ ] 没想法了，欢迎大家PR

|名称|描述|改造|测试|
|--|--|--|--|
|canny|边缘图|完成|完成|
|depth|深度图|完成|未进行|
|hough|线段识别，识别人物功能极差，非常适合建筑|已完成||未进行|
|hed|边缘检测但保留更多细节，适合重新着色和风格化|已完成|未进行|
|normal_map|根据图片生成法线贴图，非常适合CG建模师|已完成|未进行|
|openpose|人物姿势|完成|完成|
|scribble|黑白稿|已完成|未进行|
|fake_scribble|涂鸦风格|已完成|未进行|
|segmentation|分割图|已完成|未进行|
