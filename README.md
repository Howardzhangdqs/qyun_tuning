<div align="center">
<h1>Qyun Tuning 青云调参</h1>
<div>适者存千代竞逐，精微处三昧调弦</div>
</div>

## 项目描述

本项目基于遗传算法的 PID 参数优化工具，通过交互式界面帮助用户快速找到最优的 PID 控制参数组合。项目结合了 DEAP 遗传算法框架和 Gradio 可视化界面，为用户提供直观的参数优化体验。

### 主要功能

+ **交互式PID参数优化**：通过逐步提交适应度值，引导遗传算法寻找最优PID参数
+ **可视化界面**：实时显示当前参数组合、最佳参数和历史优化记录
+ **灵活的参数配置**：
  + 可调PID参数范围（Kp, Ki, Kd）
  + 可配置遗传算法参数（种群大小、迭代次数、交叉/变异概率）
  + 优化过程控制：支持随时开始或终止优化过程

## 安装指南

1. 克隆本仓库到本地：

    ```bash
    git clone https://github.com/Howardzhangdqs/qyun_tuning.git
    ```

2. 安装所需依赖：

    ```bash
    pip install -r requirements.txt
    ```

## 使用说明

运行主程序：
```bash
python app.py
```

## TODO

+ 自定义优化参数数量与名称

## 许可证
MIT License