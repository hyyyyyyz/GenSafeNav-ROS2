FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# ── 换清华 Ubuntu apt 源 ──────────────────────────────────────────────────────
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

# ── 安装基础依赖 ──────────────────────────────────────────────────────────────
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# ── 配置 locale ───────────────────────────────────────────────────────────────
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# ── 添加 ROS 2 Humble 清华源 ────────────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
         https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list

# ── 安装 ROS 2 Humble ─────────────────────────────────────────────────────────
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

# ── 初始化 rosdep（清华 rosdistro 镜像）────────────────────────────────────────
ENV ROSDISTRO_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml

RUN rosdep init && \
    sed -i 's|raw.githubusercontent.com/ros/rosdistro/master|mirrors.tuna.tsinghua.edu.cn/rosdistro|g' \
        /etc/ros/rosdep/sources.list.d/20-default.list && \
    rosdep update --rosdistro humble

# ── 工作目录 & 自动 source ────────────────────────────────────────────────────
WORKDIR /ros2_ws

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "[[ -f /ros2_ws/install/setup.bash ]] && source /ros2_ws/install/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
