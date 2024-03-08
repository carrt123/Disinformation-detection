import torch
import torchmetrics


def evaluate_accuracy(data_iter, net, device=None):  # 测试函数
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()  # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


def test_loop(data_iter, net, device=None):
    # 实例化相关metrics的计算对象
    TP, TN, FP, FN = 0, 0, 0, 0
    esp = 0.0001
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
        with torch.no_grad():
            for X, y in data_iter:
                net.eval()
                output = net(X.to(device)).argmax(dim=1).cpu()
                for i in range(y.shape[0]):
                    if output[i] == 0 and y[i] == 0:
                        TP += 1
                    if output[i] == 1 and y[i] == 1:
                        TN += 1
                    if output[i] == 0 and y[i] == 1:
                        FP += 1
                    if output[i] == 1 and y[i] == 0:
                        FN += 1
                net.train()
        P = TP / (TP + FP + esp)
        R = TP / (TP + FN + esp)
        F1 = 2 * P * R / (P + R + esp)
        acc = (TP + TN) / (TP + TN + FP + FN + esp)
        print('accuracy %.3f, precision %.3f, recall %.3f, F1 %.3f' % (acc, P, R, F1))
