import os
import numpy as np
from datetime import datetime, timedelta


def Fv6_load_meteo_data(folder_path, num_features, print_on = True):
    """
    从指定文件夹加载所有气象场数据文件，并存储为类似潮位数据的结构

    参数:
        folder_path (str): 气象数据文件夹路径
        num_features (int): 每个时间步的气象特征数量
        print_on：执行函数时，是否打印信息

    返回:
        list: 每个元素是一个站点（文件）的气象数据数组，形状为 (时间步数, num_features+1)，+1为了存储时间戳
              第一列是时间戳，后续列是气象特征值
    """
    # 获取所有.dat文件并按文件名升序排序
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.dat')])

    if not file_list:
        raise FileNotFoundError(f"在文件夹 {folder_path} 中找不到任何 .dat 文件")

    result = []  # 存储所有气象站点数据，定义空列表

    print(f"\n在文件夹 {folder_path} 找到 {len(file_list)} 个气象数据文件:")

    # 处理每个气象数据文件
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)

        # 读取文件
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f" - {file_name}: 空文件，跳过")
            continue

        # 解析文件第一行
        try:
            header_parts = lines[0].split()
            start_time_str = header_parts[0].strip()
            time_interval = int(header_parts[1])
            num_timesteps = int(header_parts[2])
        except (IndexError, ValueError) as e:
            print(f" - {file_name}: 无法解析第一行 '{lines[0].strip()}': {e}")
            continue

        # 转换开始时间为datetime对象
        try:
            time_formats = {
                14: "%Y%m%d%H%M%S",  # YYYYMMDDHHMMSS
                12: "%Y%m%d%H%M",  # YYYYMMDDHHMM
                10: "%Y%m%d%H",  # YYYYMMDDHH
                8: "%Y%m%d"  # YYYYMMDD
            }
            time_length = len(start_time_str)
            fmt = time_formats.get(time_length, "%Y%m%d%H%M%S")
            base_time = datetime.strptime(start_time_str, fmt)
        except ValueError as e:
            print(f" - {file_name}: 无法解析开始时间 '{start_time_str}': {e}")
            continue

        # 创建当前文件的时间戳列表
        file_timestamps = [base_time + timedelta(seconds=i * time_interval)
                           for i in range(num_timesteps)]

        # 读取数据部分
        data_lines = lines[1:1 + num_timesteps]  # 从第二行开始读取指定数量的行

        # 检查行数是否匹配
        if len(data_lines) < num_timesteps:
            print(f"     警告: 文件中的数据行数({len(data_lines)})少于预期的{num_timesteps}步")

        # 解析气象数据
        meteo_values = []
        valid_points = 0
        for line_idx, line in enumerate(data_lines):
            try:
                # 尝试解析一行中的所有浮点数
                values = list(map(float, line.split()))

                # 检查数据点数是否匹配
                if len(values) != num_features:
                    print(f"     行 {line_idx + 2}: 有 {len(values)} 个数据点(预期 {num_features})")

                # 添加有效数据（如果行数据不完整，添加NaN填充）
                if len(values) >= num_features:
                    meteo_values.append(values[:num_features])
                    valid_points += 1
                else:
                    # 行数据不足，用NaN填充缺失值
                    padded = values + [np.nan] * (num_features - len(values))
                    meteo_values.append(padded)
            except ValueError:
                # 解析失败，添加一行NaN
                meteo_values.append([np.nan] * num_features)

        # 创建包含时间戳和气象特征值的数组
        site_data = []
        for ts, vals in zip(file_timestamps, meteo_values):
            row = [ts.timestamp()] + vals  # 时间戳 + 气象特征值
            site_data.append(row)

        # 转换为NumPy数组并添加到结果中
        site_array = np.array(site_data)
        result.append(site_array)

        if print_on:
            # 打印文件信息
            print(f" - {file_name}:")
            print(f"     起始时间//时间步长//时间步数: {base_time.strftime('%Y-%m-%d %H:%M:%S')} // {time_interval}秒 // {num_timesteps}")
            print(f"     有效数据点//数据结构: ({valid_points}/{len(data_lines)})//{site_array.shape}")

    print(f"成功加载 {len(result)} 个气象站点的数据")
    return result