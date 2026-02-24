#!/bin/bash
set -e

cd "$(dirname "$0")"

CONTAINER_NAME="gensafenav"
IMAGE_NAME="gensafenav:humble"

# 清理模式：./build.sh -c
if [ "$1" = "-c" ]; then
    echo "清理容器和镜像..."
    docker compose down 2>/dev/null || true
    docker rmi ${IMAGE_NAME} 2>/dev/null || true
    echo "清理完成！"
    exit 0
fi

# 允许 Docker 访问 X11（rviz2 / rqt 等 GUI 工具）
xhost +local:docker > /dev/null 2>&1 || echo "警告: 无法设置 xhost，GUI 可能无法显示"

# 容器已在运行 → 直接进入
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "进入已运行的容器: ${CONTAINER_NAME}"

# 容器已存在但已停止
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "启动已停止的容器: ${CONTAINER_NAME}"
    docker compose up -d

# 容器不存在
else
    if ! docker image inspect ${IMAGE_NAME} > /dev/null 2>&1; then
        echo "构建镜像: ${IMAGE_NAME}"
        docker compose build
    fi
    echo "创建并启动容器: ${CONTAINER_NAME}"
    docker compose up -d
fi

docker exec -it ${CONTAINER_NAME} bash
