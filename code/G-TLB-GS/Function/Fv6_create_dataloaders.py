import torch
from torch.utils.data import TensorDataset, DataLoader


def Fv6_create_dataloaders(X, Y, batch_size):
    """
    创建TensorDataset和DataLoader；将numpy数组转换为torch张量，并创建TensorDataset和DataLoader。

    参数:
        X: 输入特征 (n_samples, sequence_length, n_features)
        Y: 目标值 (n_samples, output_length)
        batch_size: 批量大小
    """
    # 转换numpy数组为torch张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    # 创建数据集和数据加载器
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)   # shuffle=False，已在Fv6_prepare_sequence_data_with_meteo打乱，此处无需打乱
    return loader