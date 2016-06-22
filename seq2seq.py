#!/usr/bin/python
# coding: UTF-8
import argparse
import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers, cuda, serializers, utils
from chainer import Variable
from vocabulary import Vocabulary


# variance
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("mode", type=str, help="train mode or test mode")
    parser.add_argument("--vocabulary_size", dest="input_size", type=int, default=1000)
    parser.add_argument("--embed_size", dest="embed_size", type=int, default=1000)
    parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=1000,
                        help="the number of cells at hidden layer")
    parser.add_argument("--output_size", dest="output_size", type=int, default=1000,
                        help="the number of cells at output layer")
    parser.add_argument("--unit", dest="num_unit", type=int, default=2, help="the number of units")
    parser.add_argument("--layer", dest="num_layer", type=int, default=1, help="the number of layers in hidden layer")
    return parser.parse_args()


def text_generator(file_name):
    with open(file_name, "r") as f:
        for line in f:
            yield line
            # for文内で呼び出すと、毎回次の数を出力する。


class EncoderDecoder(chainer.Chain):
    def __init__(self, args, src_vocabulary, trg_vocabulary):
        args = parse_args()
        self.src_size = len(src_vocabulary.vocabulary)
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.trg_size = len(trg_vocabulary.vocabulary)

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


def train(data, model, source_vocabulary, target_vocabulary):
    # setup optimizer
    opt = optimizers.SGD()  # 確率勾配法
    opt.setup(model)        # 初期化

    for source_sentence, target_sentence in data:
        opt.zero_grads()
        loss = forward(model, source_sentence, target_sentence, source_vocabulary, target_vocabulary, True)
        loss.backward()            # 誤差逆伝播
        opt.clip_grads(10)         # 10より大きい勾配を抑制
        opt.update()               # パラメタ更新

    # save
    serializers.save_npz("model", model)
    serializers.save_npz("state", opt)


def test(model, source_data, source_vocabulary, target_vocabulary):
    #model = serializers.load_npz("model", EncoderDecoder(args, source_vocabulary, target_vocabulary))
    #opt = serializers.load_npz("state", optimizers.SGD())

    data = []
    for source_sentence in source_data:
        data.append(forward(model, source_sentence, None, source_vocabulary, target_vocabulary, False))
        with open("output.txt", "w") as f:
            for line in data:
                for word in line:
                    f.write(word + ' ')
                f.write("\n")
    return data


# source: English e.g. ["he", "runs", "fast", "<EOS>"]
# target: French e.g. ["il", "court", "vite", "<EOS>"]
# train: learn or generate
def forward(model, source_sentence, target_sentence, source_vocabulary, target_vocabulary, training):

    # convert word to ID, add <End of Sentence>
    source = source_vocabulary.convert(source_sentence)
    if target_sentence is not None:
        target = target_vocabulary.convert(target_sentence)

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
    EOS = target_vocabulary.word_to_id("<EOS>")
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
        print(loss.data)
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
            #exit()
            if word_id[0] == EOS:
                sentence.append(target_vocabulary.id_to_word(word_id[0]))
                break
            sentence.append(target_vocabulary.id_to_word(word_id[0]))

        print(sentence)
        return sentence


def main():
    args = parse_args()
    source_file = args.source
    v1 = Vocabulary()
    v2 = Vocabulary()
    v1.make_dictionary(source_file)
    target_file = args.target
    v2.make_dictionary(target_file)

    source_gen = text_generator(source_file)
    target_gen = text_generator(target_file)
    data_gen = zip(source_gen, target_gen)

    model = EncoderDecoder(args, v1, v2)

    if args.mode == "train":
        # 引数からモデルを定義
        train(data_gen, model, v1, v2)
    else:
        test(model, source_gen, v1, v2)



if __name__ == '__main__':
    main()

