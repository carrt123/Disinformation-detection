import json
from tkinter import *
import datetime
import tkinter.messagebox as messagebox
from 数据集处理 import get_vocab, load_data
import torch
from TextCNN模型 import TextCNN
from TextCNN_LSTM模型 import TextCNN_LSTM
from LSTM模型 import LSTM
from PIL import ImageTk, Image


class Window:
    def __init__(self, window_name):
        self.root = window_name
        self.vocab = None
        self.vocab1 = get_vocab(load_data('rumor_data.txt', 'c'))
        self.vocab2 = get_vocab(load_data('news_data.txt', 'e'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backPhoto = ImageTk.PhotoImage(Image.open('background.ppm'))
        self.max_l = 300
        self.net = None
        self.text1 = None
        self.text2 = None
        self.text3 = None
        self.canvas_root = None
        self.TimeLabel = None
        self.type = {0: "Chinese", 1: "English"}
        self.Models = {0: "LSTM        ", 1: "TextCNN     ", 2: "TextCNN_LSTM"}
        self.var1 = IntVar()
        self.var1.set(3)
        self.var2 = IntVar()
        self.var2.set(0)
        self.pred = None
        self.prob = None

    def set_init_window(self):
        self.root.title("虚假信息检测---by  Fan")
        self.root.geometry('600x400')
        self.root.resizable(width=False, height=False)
        self.canvas_root = Canvas(self.root, width=600, height=400, highlightthickness=0)
        self.canvas_root.create_image(300, 200, image=self.backPhoto)
        self.canvas_root.pack()
        #
        self.text1 = Text(self.root, width=42, height=15, font=('宋体', 10))
        self.text1.place(x=0, y=0)
        self.text2 = self.canvas_root.create_text(450, 20, fill='orangered',text="欢迎使用虚假信息检测工具", font=('楷体', 18,'bold'))
        # 基本功能按钮
        Button(self.root, text="获取结果", command=self.getOutcome, relief=RAISED,
               bg='Gold', font=('楷体', 12)).place(x=0, y=210)
        Button(self.root, text="清空输入", command=self.clean, relief=RAISED,
               bg="DarkOrange", font=('楷体', 12)).place(x=80, y=210)
        Button(self.root, text="点击退出", command=self.exit, relief=RAISED,
               bg="Bisque", font=('楷体', 12)).place(x=160, y=210)
        Button(self.root, text="保存数据", command=self.save_data, font=('楷体', 12)).place(x=240, y=210)
        # 模型功能按钮
        for val, t in self.type.items():
            Radiobutton(self.root, text=t, variable=self.var2, value=val, bg='orange').place(x=val * 80, y=265)
        for val, model in self.Models.items():
            Radiobutton(self.root, text=model, variable=self.var1, value=val, command=self.load_net, bg='orange',
                        font=('楷体', 10,)).place(y=300 + val * 25)
        self.TimeLabel = self.canvas_root.create_text(450, 150, fill='orange',font=('楷体', 18, 'bold'),
                                                      text="%s" % (
                                                          datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.root.after(100, self.uptime())

    def error_message(self):
        messagebox.showinfo(title='提示', message="未输入任何信息，请输入信息后使用", parent=self.root)

    def uptime(self):
        self.canvas_root.itemconfig(self.TimeLabel, text="%s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.root.after(100, self.uptime)

    def load_net(self):
        lang = self.type[self.var2.get()]
        model = self.Models[self.var1.get()]
        if lang == 'Chinese':
            self.vocab = self.vocab1
            if model == 'TextCNN     ':
                self.net = TextCNN(len(self.vocab1))
                self.net.load_state_dict(torch.load('signTextCNNNet.pth'))
            elif model == 'LSTM        ':
                self.net = LSTM(len(self.vocab1))
                self.net.load_state_dict(torch.load('signLSTMNet.pth'))
            elif model == 'TextCNN_LSTM':
                self.net = TextCNN_LSTM(len(self.vocab1))
                self.net.load_state_dict(torch.load('signTextCNN_LSTMNet.pth'))

        else:
            self.vocab = self.vocab2
            if model == 'TextCNN     ':
                self.net = TextCNN(len(self.vocab2))
                self.net.load_state_dict(torch.load('#2gloveTextCNNNet.pth'))
            elif model == 'LSTM        ':
                self.net = LSTM(len(self.vocab2))
                self.net.load_state_dict(torch.load('gloveLSTMNet.pth'))
            elif model == 'TextCNN_LSTM':
                self.net = TextCNN_LSTM(len(self.vocab2))
                self.net.load_state_dict(torch.load('#2gloveTextCNN_LSTMNet.pth'))

        messagebox.showinfo(title='提示', message="当前你使用的是" + model.strip() + "模型", parent=self.root)
        self.net.to(self.device)

    def pad(self, x):
        return x[:self.max_l] if len(x) > self.max_l else x + [0] * (self.max_l - len(x))

    def predict(self, text):
        self.net.eval()
        words = [x for x in text]

        with torch.no_grad():
            X = torch.tensor(self.pad([self.vocab.stoi[word] for word in words]))
            X = torch.unsqueeze(X, 0)

            output = self.net(X.to(self.device))
            self.prob = torch.softmax(output, dim=1)
            self.pred = torch.argmax(self.prob, dim=1)

            if self.pred.item() == 1:
                messagebox.showinfo(title='提示', message="这是条真实信息，置信度评分为：{}".format(self.prob[0][1]), parent=self.root)
            else:
                messagebox.showinfo(title='提示', message="这是条虚假信息，置信度评分为：{}".format(self.prob[0][0]), parent=self.root)

    def getOutcome(self):
        text = self.text1.get(1.0, 'end')
        if self.var1.get() > 2:
            messagebox.showinfo(title='提示', message="请先选择模型", parent=self.root)
        else:
            if len(text) > 1:
                self.predict(text)
            else:
                self.error_message()

    def save_data(self):
        # 获取文本框中的内容
        text = self.text1.get('1.0', 'end')
        # 获取时间戳并格式化
        curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        data = {"time": curtime, "text": text, "net": self.Models[self.var1.get()].strip(),
                "probability": self.prob[0][self.pred.item()].tolist(), "label": self.pred.item()}
        # 打开文件并追加文本和时间戳
        with open('text_file.json', 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

    def clean(self):
        self.text1.delete(1.0, 'end')

    def exit(self):
        if messagebox.askyesno("Quit", "你确定想要退出吗?"):
            self.root.destroy()


if __name__ == '__main__':
    root = Tk()
    tk = Window(root)
    tk.set_init_window()
    tk.root.mainloop()
