#!/usr/bin/python
# coding: UTF-8
import argparse
import numpy as np

import chainer
import chainer.functions as F
from chainer import optimizers, serializers
from chainer import Variable

from vocabulary import Vocabulary


# variance
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("mode", type=str, help="train mode or test mode")
    parser.add_argument("--epoch", dest="epoch", type=int, default=10)
    parser.add_argument("--embed_size", dest="embed_size", type=int, default=100)
    parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=200,
                        help="the number of cells at hidden layer")
    return parser.parse_args()


def text_generator(file_name):
    with open(file_name, "r") as f:
        for line in f:
            print(line)
            yield line


class EncoderDecoder(chainer.Chain):
    def __init__(self, args, src_file, trg_file):

        self.src_vocabulary = Vocabulary()
        self.src_vocabulary.make_dictionary(src_file)

        self.trg_vocabulary = Vocabulary()
        self.trg_vocabulary.make_dictionary(trg_file)

        self.src_size = len(self.src_vocabulary.wtoi)
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.trg_size = len(self.trg_vocabulary.wtoi)

        super(EncoderDecoder, self).__init__(
            # encoder
            w_xe=F.EmbedID(self.src_size, self.embed_size),
            w_ep=F.Linear(self.embed_size, self.hidden_size*4),
            w_pp=F.Linear(self.hidden_size, self.hidden_size*4),
            # decoder
            w_ye=F.EmbedID(self.trg_size, self.embed_size),
            w_eq=F.Linear(self.embed_size, self.hidden_size*4),
            w_qq=F.Linear(self.hidden_size, self.hidden_size*4),
            w_qy=F.Linear(self.hidden_size, self.trg_size),
        )


def train(args, model):

    # setup optimizer
    opt = optimizers.SGD()  # 確率勾配法
    opt.setup(model)        # 初期化

    for i in range(args.epoch):
        src_generator = text_generator(args.source)
        trg_generator = text_generator(args.target)

        total_loss = 0.0
        for src_sentence, trg_sentence in zip(src_generator, trg_generator):
            opt.zero_grads()
            loss = forward(model, src_sentence, trg_sentence, True)
            total_loss += loss.data
            loss.backward()            # 誤差逆伝播
            opt.clip_grads(10)         # 10より大きい勾配を抑制
            opt.update()               # パラメタ更新
        print("epoch: %3d, loss: %f" % (i, total_loss))

    # save
    serializers.save_npz("model", model)
    # TODO : save args
    #serializers.save_npz("state", opt)


def test(args, model):
    #serializers.load_npz("model", model)
    #opt = serializers.load_npz("state", optimizers.SGD())

    data = []
    for source_sentence in text_generator(args.source):
        data.append(forward(model, source_sentence, None, False))
        with open("output.txt", "w") as f:
            for line in data:
                for word in line:
                    f.write(word + ' ')
                f.write("\n")
    return data


def forward(model, source_sentence, target_sentence, training):

    # convert word to ID, add <End of Sentence>
    source = model.src_vocabulary.convert(source_sentence)
    if target_sentence:
        target = model.trg_vocabulary.convert(target_sentence)

    # hidden state
    c = Variable(np.zeros((1, model.hidden_size), dtype=np.float32))  # hidden state
    p = Variable(np.zeros((1, model.hidden_size), dtype=np.float32))

    # encoder
    for word_id in source[::-1]:
        x = Variable(np.array(word_id, dtype=np.int32))  #input one-hot vector
        e = model.w_xe(x)   # 単語ベクトル
        p1 = model.w_ep(e)           # hidden layer input
        p2 = model.w_pp(p)

        # TODO: W * h
        lstm_input = p1 + p2
        # array ( W * x + W * h ) sigmoid のなかみ
        c, p = F.lstm(c, lstm_input)

    # decoder
    EOS = model.trg_vocabulary.word_to_id("<EOS>")
    q = p
    y = Variable(np.array([EOS], dtype=np.int32))

    if training:
        loss = Variable(np.zeros((), dtype=np.float32))
        for word_id in target:
            e = model.w_ye(y)
            lstm_input = model.w_qq(q) + model.w_eq(e)
            c, q = F.lstm(c, lstm_input)
            y = model.w_qy(q)
            t = Variable(np.array(word_id, dtype=np.int32))
            loss += F.softmax_cross_entropy(y, t)
            y = t
        #print(loss.data)
        return loss

    else:
        sentence = []
        while len(sentence) < 100:
            e = model.w_ye(y)
            lstm_input = model.w_qq(q) + model.w_eq(e)
            c, q = F.lstm(c, lstm_input)
            y = model.w_qy(q)
            word_id = np.argmax(y.data, axis=1)
            y = Variable(np.array(word_id, dtype=np.int32))
            if word_id[0] == EOS:
                sentence.append(model.trg_vocabulary.id_to_word(word_id[0]))
                break
            sentence.append(model.trg_vocabulary.id_to_word(word_id[0]))

        print(sentence)
        return sentence


def main():
    args = parse_args()
    model = EncoderDecoder(args, args.source, args.target)

    if args.mode == "train":
        # 引数からモデルを定義
        train(args, model)
    else:
        # TODO : load args
        serializers.load_npz("model", model)
        test(args, model)


if __name__ == '__main__':
    main()

