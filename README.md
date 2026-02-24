# GenSafeNav-ROS2

## 简介

该仓库是用于在真实机器人上部署 GenSafeNav 策略的 ROS2 系统。它集成了行人检测、跟踪、轨迹预测和基于强化学习的决策制定，以实现安全的人群导航。

## 项目结构

```
.
├── decider/                 # 基于强化学习（RL）的决策模块
│   ├── decider/             # ROS2 主节点
│   ├── rl/networks/         # 策略网络 (selfAttn_srnn 架构)
│   ├── config/              # 配置文件
│   └── model_weight/        # 预训练模型权重 (ours.pt)
├── predictor/               # 轨迹预测模块
│   ├── predictor/           # 集成 DtACI 的 ROS2 主节点
│   └── gst_updated/         # Gumbel Social Transformer 模型
├── dr_spaam_ros2/           # 2D 激光雷达行人检测 (DR-SPAAM)
├── sort_tracker/            # 多目标追踪 (SORT)
├── command_listener/        # 用户指令接口
├── frequency_monitor/       # 系统性能监控
├── fake_detection/          # 仿真测试工具
├── docker/                  # docker启动文件
├── FAST_LIO/                # 里程计模块
├── gensafenav_ros2_bringup  # 项目总启动文件
├── livox_ros_driver2        # mid-360雷达驱动模块
└── pointcloud_to_laserscan  # 点云格式转换模块
```

## 简单开始

本项目基于 docker 构建运行，由于镜像限制，目前不适用于50系列显卡（因为架构的改变），目前环境只在 RTX 4070 上测试，docekr 容器中安装的 ROS2 版本为 humble。

### 环境准备

```bash
git clone --recursive https://github.com/hyyyyyyz/GenSafeNav-ROS2.git
cd GenSafeNav-ROS2/docker
sudo chmod +x ./build.sh
sudo ./build
```

### 项目编译

```bash
colcon build --symlink-install
```

### 运行

```bash
source install/setup.bash
ros2 launch gensafenav_ros2_bringup gensafenav_ros2_bringup.launch.py 
```

## 致谢
本项目是基于[GenSafeNav-ROS2](https://github.com/tasl-lab/GenSafeNav-ROS2)的二次开发，主要方便大家在目前通用的 ROS2-humble 版本上进行开发。