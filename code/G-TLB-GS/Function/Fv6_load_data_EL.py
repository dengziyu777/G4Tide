import os
import numpy as np
from datetime import datetime, timedelta

def Fv6_load_data_EL(file_path, scale_factor, adjust, print_on = True):
    """
    加载新格式的数据文件（单文件格式）；列排序时先时间步数多站点、NaN值提前处理

    数据格式：
        第一行：点位数量N（整数）
        第2行到第N+1行：各点位数据的开始时间（YYYYMMDDHHMMSS）、时间间隔（秒）、时间步数
        第N+2行开始：潮位数据矩阵，行数为所有站点中最大时间步数，每行包含各点位的潮位数据
        - 文件中已为时间步数较少的站点后续位置填充NaN
    参数:
        file_path: 数据文件路径
        scale_factor: 数据缩放系数
        adjust: 基准面调整值（米）
        print_on：执行函数时，是否打印信息

    返回:
        list: 包含每个站点数据的Python列表，结构为:
          [
            site1_array,  # 站点1的二维NumPy数组
            site2_array,  # 站点2的二维NumPy数组
            ...           # 其他站点的数据数组
          ]

        每个站点的数据结构:
          二维NumPy数组，形状为 (M, 2)，其中：
            M = 该站点的实际数据点数
            列0: float类型的时间戳（UNIX时间戳，UTC时间）
            列1: float类型的潮位值（经过缩放和调整，单位：米）

          示例结构:
            array([
                [timestamp1, value1],
                [timestamp2, value2],
                ...
            ])

        使用示例:
          获取所有站点列表: sites_data = Fv6_load_data_EL(...)
          获取第1个站点: site1 = sites_data[0]
          获取时间戳列: timestamps = site1[:, 0]
          获取潮位值列: water_levels = site1[:, 1]

          将时间戳转换为可读时间:
            from datetime import datetime
            dt = datetime.utcfromtimestamp(timestamp)
    """
    # 读取整个文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 解析第一行获取站点数量
    try:
        num_sites = int(lines[0])
    except ValueError:
        print(f"错误: 无法解析站点数量 - {lines[0]}")
        return []
    site_info = []
    if print_on:
        print(f"\n解析{file_path}信息:")
        print(f"{'站点':<5}{'开始时间':<25}{'时间间隔(秒)':<15}{'时间步数':<15}")

    # 解析每个站点的信息
    for i in range(1, num_sites + 1):
        parts = lines[i].split()
        if len(parts) < 3:
            print(f"错误: 第{i + 1}行数据格式不正确: {lines[i]}")
            continue

        start_time_str = parts[0].strip()
        try:
            time_interval = int(parts[1])
            time_steps = int(parts[2])
        except ValueError as e:
            print(f"错误: 第{i + 1}行解析数字失败: {parts[1]} 或 {parts[2]} - {e}")
            continue

        # 解析时间格式
        try:
            time_formats = {
                14: "%Y%m%d%H%M%S",  # YYYYMMDDHHMMSS
                12: "%Y%m%d%H%M",  # YYYYMMDDHHMM
                10: "%Y%m%d%H",  # YYYYMMDDHH
                8: "%Y%m%d"  # YYYYMMDD
            }
            time_length = len(start_time_str)
            fmt = time_formats.get(time_length, "%Y%m%d%H%M%S")
            start_time = datetime.strptime(start_time_str, fmt)
        except ValueError as e:
            print(f"错误: 解析时间 '{start_time_str}' 失败 - {e}")
            start_time = datetime(1970, 1, 1)  # 使用默认时间

        site_info.append({
            'start_time': start_time,
            'time_interval': time_interval,
            'time_steps': time_steps
        })

        if print_on:
            # 打印站点信息
            start_time_fmt = start_time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i:<5}{start_time_fmt:<25}{time_interval:<15}{time_steps:<15}")

    # 获取最大时间步数
    max_time_steps  = max([info['time_steps'] for info in site_info if info is not None])

    # 确定数据起始行（观测数据的行数）
    data_start_line = num_sites + 1     # Python中索引从0开始
    if len(lines) < data_start_line:
        print(f"错误: 文件行数不足，缺少数据部分")
        return []
    if (len(lines) - data_start_line) != max_time_steps:
        print(f"警告: 数据行数({len(lines)-data_start_line}) ≠ 最大时间步数({max_time_steps})")

    # 读取数据部分（所有站点的潮位数据）
    data_lines = lines[data_start_line:]

    # 创建数据结构：每个站点一个列表
    sites_data = [[] for _ in range(num_sites)]
    sites_timestamps = [[] for _ in range(num_sites)]

    # 创建所有站点的当前时间
    current_times = [info['start_time'] for info in site_info]

    # print(f"\n处理数据矩阵: 共有 {len(data_lines)} 行数据")

    # 处理每一行数据（每个时间步）
    for row_index in range(max_time_steps):
        # 检查当前行是否存在
        if row_index >= len(data_lines):
            # 文件行不足，用NaN填充所有站点
            for site_idx in range(num_sites):
                if row_index < site_info[site_idx]['time_steps']:
                    sites_data[site_idx].append(np.nan)
                    sites_timestamps[site_idx].append(current_times[site_idx].timestamp())
            continue

        line = data_lines[row_index]
        parts = line.split()

        # 保证每行数据列数等于站点数（不足用'NaN'填充，过多则截断）
        if len(parts) < num_sites:
            parts += ['NaN'] * (num_sites - len(parts))
        elif len(parts) > num_sites:
            parts = parts[:num_sites]

        # 为每个站点处理数据
        for site_idx in range(num_sites):
            # 跳过已经完成所有时间步的站点
            if row_index >= site_info[site_idx]['time_steps']:
                continue

            # 处理当前站点的数据
            value_str = parts[site_idx].strip()

            # 处理NaN值和无效值
            if value_str.upper() in ('NAN', 'NA', 'NULL', ''):
                sites_data[site_idx].append(np.nan)
            else:
                try:
                    raw_value = float(value_str)
                    adjusted_value = raw_value * scale_factor + adjust
                    sites_data[site_idx].append(adjusted_value)
                except ValueError:
                    sites_data[site_idx].append(np.nan)

            # 添加时间戳
            sites_timestamps[site_idx].append(current_times[site_idx].timestamp())

            # 更新时间（仅未完成时间步的站点）
            current_times[site_idx] += timedelta(seconds=site_info[site_idx]['time_interval'])

    # 创建最终数据结构
    result = []
    if print_on:
        print(f"\n{file_path}数据汇总:")
        print(f"{'站点':<5}{'起始时间':<25}{'时间步长(秒)':<15}{'预期时间步数':<15}{'实际加载点数':<15}{'完成度':<15}{'缺失点':<10}")

    for site_idx in range(num_sites):
        if not sites_timestamps[site_idx]:
            # 创建空数组
            site_array = np.empty((0, 2))
        else:
            site_array = np.column_stack((
                sites_timestamps[site_idx],
                sites_data[site_idx]
            ))
        result.append(site_array)

        # 获取站点信息
        info = site_info[site_idx]
        expected_steps = info['time_steps']
        actual_steps = len(sites_data[site_idx])

        # 计算缺失点
        missed_steps = expected_steps - actual_steps
        missed_percent = (missed_steps / expected_steps) * 100 if expected_steps > 0 else 0

        if print_on:
            # 计算完成度
            completeness = "100%" if actual_steps == expected_steps else f"{actual_steps / expected_steps * 100:.1f}%"
            # 格式化输出信息
            start_time_str = info['start_time'].strftime("%Y-%m-%d %H:%M:%S")
            # 显示站点信息
            print(f"{site_idx + 1:<5}{start_time_str:<25}{info['time_interval']:<15}{expected_steps:<15}{actual_steps:<15}{completeness:<15}{missed_steps:<10}")

        # 检查数据完整性
        if missed_steps > 0:
            print(f"! 警告: 站点{site_idx + 1}缺少{missed_steps}个数据点 ({missed_percent:.1f}%)")

    total_points = sum(len(s) for s in result)
    expected_total = sum(info['time_steps'] for info in site_info)
    total_missed = expected_total - total_points
    print(f"数据加载完成：共{num_sites}个站点")
    if print_on:
        print(f"总数据点数量: {total_points}/{expected_total} (缺失: {total_missed}, {total_missed / expected_total * 100:.1f}%)")

    return result