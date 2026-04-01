import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def Fv6_generate_time_ticks(start_timestamp, end_timestamp, interval_hours):
    """
    生成时间刻度及其标签（确保包含最后一个时间点）

    参数:
        start_timestamp: 开始时间戳
        end_timestamp: 结束时间戳
        interval_hours: 时间间隔(小时)

    返回:
        tick_locations: 刻度位置(时间戳列表)
        tick_labels: 刻度标签(字符串列表)
    """
    start_dt = datetime.fromtimestamp(start_timestamp)
    end_dt = datetime.fromtimestamp(end_timestamp)

    # 对齐到最近的整点
    current_dt = start_dt.replace(minute=0, second=0, microsecond=0)
    if current_dt < start_dt:
        current_dt += timedelta(hours=1)

    tick_locations = []
    tick_labels = []

    # 生成常规间隔的刻度
    while current_dt <= end_dt:
        tick_locations.append(current_dt.timestamp())
        tick_labels.append(current_dt.strftime('%Y-%m-%d %H:%M'))
        current_dt += timedelta(hours=interval_hours)

    # 确保最后一个时间点被包含
    if len(tick_locations) == 0 or tick_locations[-1] != end_timestamp:
        # 如果最后一个刻度不等于结束时间，则添加结束时间
        tick_locations.append(end_timestamp)
        tick_labels.append(end_dt.strftime('%Y-%m-%d %H:%M'))

    # 对时间戳进行排序（以防间隔导致顺序错乱）
    tick_locations, tick_labels = zip(*sorted(zip(tick_locations, tick_labels)))

    return list(tick_locations), list(tick_labels)
