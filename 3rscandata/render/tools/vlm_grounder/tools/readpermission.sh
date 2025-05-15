# 创建目标目录并复制所有文件（保留路径结构）
while IFS= read -r src_path; do
    dst_path="humiao/${src_path#/}"          # 移除路径开头的/
    sudo mkdir -p "$(dirname "$dst_path")"   # 递归创建目录
    sudo cp --preserve=all "$src_path" "$dst_path"
    echo "Copied: $src_path → $dst_path"
done < tupianoutput.txt