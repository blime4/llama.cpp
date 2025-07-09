#!/bin/bash
set -e
export PATH="$PATH:$HOME/.local/bin"
echo "Hostname: $(hostname)"
mkdocs --version
mike --version

git config --global user.name "Xiwen Xing"
git config --global user.email "xiwen.xing@denglin.ai"

# 获取所有包含dl-的tag并排序
echo "获取包含dl-的tag..."
ALL_TAGS=$(git for-each-ref --sort=creatordate --format='%(creatordate:iso8601) %(refname:short)' refs/tags | 
           grep "dl-" | 
           awk '$1 >= "2025-04-14" {print $NF}') # 筛选2025-03-01之后创建的tag
echo "符合条件的tag: ${ALL_TAGS}"

# 安全删除本地gh-pages分支(如果存在)
git branch -D gh-pages 2>/dev/null || true

# 创建数组存储成功部署的tag
DEPLOYED_TAGS=()

# 为每个符合条件的tag部署文档
if [ -n "$ALL_TAGS" ]; then
    # 使用所有2025-03-01后创建的包含dl-的tag
    RECENT_TAGS="$ALL_TAGS"
    for tag in $RECENT_TAGS; do
        echo "部署tag: $tag"
        git checkout $tag
# 检查当前分支上是否存在mkdocs.yml文件
        if [ -f "mkdocs.yml" ]; then
        # 部署到public目录前缀下的gh-pages分支
        mike deploy --deploy-prefix public $tag
# 记录成功部署的tag
            DEPLOYED_TAGS+=("$tag")
else
            echo "警告: 在 $tag 上未找到 mkdocs.yml 文件，跳过该tag的文档部署"
        fi
    done
    
    # 部署主分支为latest版本
    git checkout ${CI_COMMIT_REF_NAME}
    echo "部署当前分支为latest版本..."
if [ -f "mkdocs.yml" ]; then
    mike deploy --deploy-prefix public latest
    
    # 设置最新tag为默认版本
    if [ ${#DEPLOYED_TAGS[@]} -gt 0 ]; then
        LATEST_TAG=${DEPLOYED_TAGS[${#DEPLOYED_TAGS[@]}-1]}
        echo "设置最新tag $LATEST_TAG 为默认版本..."
        mike set-default --deploy-prefix public $LATEST_TAG
    else
        echo "没有可设置为默认版本的已部署tag，将latest设为默认..."
        mike set-default --deploy-prefix public latest
    fi
else
echo "警告: 在当前分支上未找到 mkdocs.yml 文件，跳过latest版本的文档部署"
    fi
else
    # 如果没有包含dl-的tag，只部署当前分支为latest版本
    echo "没有找到包含dl-的tag，部署当前分支为latest版本..."
    mike deploy --deploy-prefix public latest
    mike set-default --deploy-prefix public latest
fi

# 从gh-pages分支检出public目录
echo "从gh-pages分支检出public目录..."
git checkout gh-pages -- public

# 打印调试信息
echo "public目录内容:"
ls -la public/

echo "部署完成，文档已生成在public/目录中"