import collections
import random
import torch
import torchtext.vocab as Vocab
import re


def load_data(file_path, phase=None):
    data = []
    mx, mn, avg_num, num = 0, 100000, 0, 0
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if phase == 'c':
            for line in lines:
                sentence = line.strip().split('\t')[1]
                words = [x for x in sentence]
                num += 1
                # if len(words) < 5:
                #     continue
                # if len(words) > mx:
                #     mx = len(words)
                # if len(words) < mn:
                #     mn = len(words)
                avg_num += len(words)
                label = line.strip().split('\t')[0]
                data.append([words, int(label)])
        elif phase == 'e':
            for line in lines:
                sentence = line.split('\t')[1]
                sentence = clean_text(sentence)
                words = sentence.split(' ')
                num += 1
                if len(words) < 2:
                    continue
                # if len(words) < 5:
                #     continue
                # if len(words) > mx:
                #     mx = len(words)
                # if len(words) < mn:
                #     mn = len(words)
                avg_num += len(words)
                label = line.strip().split('\t')[0]
                data.append([words, int(label)])
        else:
            print('phase格式不正确')
            exit()
        # print('句子最大长度{}, 句子最短长度{}, 句子平均长度{}'.format(mx, mn, int(avg_num / num)))
        random.shuffle(data)
    return data


def get_vocab(data):
    tokenized_data = [word for word, _ in data]
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter)


def preprocess(data, vocab, max_l=300):
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = [word for word, _ in data]
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


def clean_text(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)

    # spelling correction
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" fb ", " facebook ", text)
    text = re.sub(r"facebooks", " facebook ", text)
    text = re.sub(r"facebooking", " facebook ", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r"dollars", " dollar ", text)

    # punctuation
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"\\", " \ ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"&", " & ", text)
    text = re.sub(r"\|", " | ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ( ", text)

    # symbol replacement
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)

    # remove extra space
    text = ' '.join(text.split())

    return text
