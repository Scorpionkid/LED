# 文件名: fix_paths.py

def process_file(input_file, output_file=None):
    """
    处理训练列表文件，移除路径中的./Sony/前缀

    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    if output_file is None:
        output_file = input_file

    # 读取原始文件
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 处理每一行
    new_lines = []
    for line in lines:
        # 替换路径前缀
        new_line = line.replace('./Sony/short/', './short/').replace('./Sony/long/', './long/')
        new_lines.append(new_line)

    # 写入新文件
    with open(output_file, 'w') as f:
        f.writelines(new_lines)

    print(f"处理完成! 已将 {len(lines)} 行从 {input_file} 写入到 {output_file}")

if __name__ == "__main__":
    input_file = "datasets/txtfiles/SID/SonyA7S2/Sony_train_list.txt"

    # 如果你想保留原文件，可以指定一个新的输出文件名
    # output_file = "datasets/txtfiles/SID/SonyA7S2/Sony_train_list_fixed.txt"
    # process_file(input_file, output_file)

    # 直接覆盖原文件
    process_file(input_file)