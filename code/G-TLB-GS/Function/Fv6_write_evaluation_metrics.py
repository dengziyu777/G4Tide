import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def Fv6_write_evaluation_metrics_part6(X_list, Y_list, output_file, note):
    """
    计算并写入评估指标到文件 - 支持每个站点独立数据结构

    参数:
        X_list: 预测值列表，每个元素是一个站点的预测值数组
        Y_list: 实际值列表，每个元素是一个站点的实际值数组
        output_file: 输出文件路径
        note: 评估说明
    """
    with open(output_file, 'a', encoding='utf-8') as f:
        # 写入评估说明
        f.write(f"\n{note}\n")
        f.write("Site\tMAE\tRMSE\tR²\tSampleCount\n")

        # 遍历每个站点
        for site_idx, (X, Y) in enumerate(zip(X_list, Y_list)):
            # 确保数据长度一致
            if len(X) != len(Y):
                print(f"站点 {site_idx + 1}: 预测值和实际值长度不一致 ({len(X)} vs {len(Y)}), 使用较短的长度")
                min_len = min(len(X), len(Y))
                X = X[:min_len]
                Y = Y[:min_len]

            # 计算当前站点指标
            mae = mean_absolute_error(Y, X)
            rmse = np.sqrt(mean_squared_error(Y, X))
            r2 = r2_score(Y, X)
            sample_count = len(Y)

            # 写入结果
            f.write(f"{site_idx + 1}\t{mae:.4f}\t{rmse:.4f}\t{r2:.4f}\t{sample_count}\n")