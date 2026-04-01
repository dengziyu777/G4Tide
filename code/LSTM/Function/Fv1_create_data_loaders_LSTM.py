from torch.utils.data import DataLoader, TensorDataset


def Fv1_create_data_loaders_LSTM(preprocessed_data, batch_size):
    """
    创建训练、验证和测试数据加载器
    """
    # 从预处理数据中提取张量
    X_train = preprocessed_data['X_train']
    y_train = preprocessed_data['y_train']
    X_val = preprocessed_data['X_val']
    y_val = preprocessed_data['y_val']
    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']

    # 创建数据集
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"数据加载器创建完成:")
    print(f"训练集: {len(train_loader)}批次, 每批{batch_size}个样本")
    print(f"验证集: {len(val_loader)}批次")
    print(f"测试集: {len(test_loader)}批次")

    return train_loader, val_loader, test_loader