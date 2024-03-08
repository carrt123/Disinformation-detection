from 数据集处理 import load_data, get_vocab, preprocess
from LSTM模型 import LSTM
from TextCNN模型 import TextCNN
from TextCNN_LSTM模型 import TextCNN_LSTM
from 加载预训练词向量 import load_pretrained_embedding
from 模型评估 import evaluate_accuracy, test_loop
from 可视化 import draw_train_process
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import os
import torchtext.vocab as Vocab
import torch
import time
from torch import nn
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):  # 训练函数
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    train_loss, train_epoch, train_acc = [], [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in tqdm(train_iter):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        train_loss.append(train_l_sum / batch_count)
        train_epoch.append(epoch)
        train_acc.append(train_acc_sum / n)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    test_loop(test_iter, net)
    draw_train_process(train_epoch, train_loss, train_acc)


def model(phase, vocab_size):
    if phase == 'TextCNN':
        return TextCNN(vocab_size, embedding_dim=300, hidden_dim=300)
    elif phase == 'LSTM':
        return LSTM(vocab_size)
    elif phase == 'TextCNN_LSTM':
        return TextCNN_LSTM(vocab_size)
    else:
        print('没有这个模型')
        exit()


if __name__ == '__main__':
    cdata_path = 'rumor_data.txt'
    edata_path = 'News_data.txt'
    train_data, test_data = train_test_split(load_data(edata_path, 'e'), test_size=0.2)
    # vocab = get_vocab(load_data(cdata_path, 'c'))
    vocab = get_vocab(load_data(edata_path, 'e'))
    batch_size = 32
    train_set = Data.TensorDataset(*preprocess(train_data, vocab, 2000))
    test_set = Data.TensorDataset(*preprocess(test_data, vocab, 2000))
    # train_set = Data.TensorDataset(*preprocess(train_data, vocab, 300))
    # test_set = Data.TensorDataset(*preprocess(test_data, vocab, 300))
    train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = Data.DataLoader(test_set, batch_size)

    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    # glove_vocab = Vocab.Vectors(name='./sgns.weibo.bigram-char', cache=cache)
    glove_vocab = Vocab.GloVe(name='6B', dim=300, cache=cache)
    name = 'TextCNN_LSTM'
    net = model(name, len(vocab))
    net.dynamic_embedding.weight.data.copy_(
        load_pretrained_embedding(vocab.itos, glove_vocab))
    net.dynamic_embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它
    lr, num_epochs = 0.0003, 20
    # 要过滤掉不计算梯度的embedding参数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    loss = nn.CrossEntropyLoss()
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

    torch.save(net.state_dict(), '#2glove' + name + 'Net.pth')
    # torch.save(net.state_dict(), 'sign' + name + 'Net.pth')
