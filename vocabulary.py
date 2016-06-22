import sys


class Vocabulary:
    def __init__(self):
        self.vocabulary = {"<unk>": 0, "<EOS>": 1}
        self.id_list = {0: "<unk>", 1: "<EOS>"}

    def make_dictionary(self, input_file):
        with open(input_file, "r") as f:
            for line in f:
                for word in line.strip().split():
                    if word not in self.vocabulary:
                        self.vocabulary[word] = len(self.vocabulary)
                        self.id_list[len(self.vocabulary) - 1] = word
        return self.vocabulary, self.id_list

    def word_to_id(self, word):
        if word in self.vocabulary:
            return self.vocabulary[word]
        else:
            return self.vocabulary["<unk>"]

    def convert(self, sentence):
        id_list =[]
        for word in sentence.split():
            id_list.append([self.word_to_id(word)])
        if id_list[-1] == self.vocabulary['<EOS>']:
            return id_list
        return id_list + [[self.vocabulary['<EOS>']]]

    def id_to_word(self, word_id):
        if word_id not in self.id_list:
            return '<unk>'
        return self.id_list[word_id]

    def revert(self, word_ids):
        word_list =[]
        for word_id in word_ids:
            word_list.append([self.id_to_word(word_id)])
        return word_list


def main(input_file):
    text = Vocabulary()
    text.make_dictionary(input_file)

if __name__ == '__main__':
    main(input_file=sys.argv[1])
