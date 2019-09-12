#!/usr/bin/env python36
# -*- coding:utf-8 -*-
# @Time    : 19-8-9 下午3:24
# @Author  : Xinxin Zhang
"""
this is an example to explain how we can use the class of DataPreprocessing and Dataloader


data {  'WORD':{'text':[['中国', '去年', '进出口', '总值', '逾', '三千二百五十亿', '美元'],
                        ...,
                        ['（', '完', '）'],   ]         max sentence :[word level:242, char level:409]
                       [[],
                        ...,
                        ['（', '完', '）']    ]
                        ...                            max text = 1809
                'text2index':
                'vocab':
                'vocab2index':          }
        'PARSING':
        'POS':
        'NER':
        'char_leval_data':
        }
"""
import os
import json
from tools.utils import Voc
from base.data.base_data_preprocessor import DataPreprocessor
from base.data.base_data_loader import Dataloader


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ this is an implement example ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OntonotesDataProcess(DataPreprocessor):
    def __init__(
            self,
            name,
            raw_data_path,
            vocab_path,
            model_data_path,
            labeled=True,
    ):
        super().__init__(name, raw_data_path, vocab_path, model_data_path,
                         labeled)
        # super().__init__(name=name,
        #                  raw_data_path=raw_data_path,
        #                  vocab_path=vocab_path,
        #                  model_data_path=model_data_path,
        #                  labeled=labeled)
        self.name = name
        self.raw_data_path = raw_data_path
        self.vocab_path = vocab_path.format(self.name) + '/{}_vocab.txt'
        # self.vocab_path = vocab_path+'/{}_vocab.txt'
        self.model_data_path = model_data_path.format(self.name)
        self.labeld = labeled
        # bert_model_dir = 'bert-base-chinese'
        # self.tokenizer = BertTokenizer.from_pretrained( 'bert-base-chinese',
        #                                                 do_lower_case=True)

        self.raw_data = {}
        self.char_level_data = {
            'WORD': {
                'text': [],
                'text2index': [],
                'vocab': {},
                'vocab2index': {}
            },
            'NER': {
                'text': [],
                'text2index': [],
                'vocab': {},
                'vocab2index': {}
            },
            'SEG': {
                'text': [],
                'text2index': [],
                'vocab': {},
                'vocab2index': {}
            }
        }

        self.dataloader_format = []

        self.WORD = {
            'text': [],
            'text2index': [],
            'vocab': {},
            'vocab2index': {}
        }
        self.POS = {
            'text': [],
            'text2index': [],
            'vocab': {},
            'vocab2index': {}
        }
        self.PARSING = {
            'text': [],
            'text2index': [],
            'vocab': {},
            'vocab2index': {}
        }
        self.NER = {
            'text': [],
            'text2index': [],
            'vocab': {},
            'vocab2index': {}
        }

    def initAttr(self):
        if os.path.isfile(self.model_data_path):
            # load all data
            self.loadData()
        else:
            # get raw data
            self.readRawData()
            # build char level data
            self.charLevelProcess()
            # build vocabulary
            self.buildVocabulary()
            # convert data to index
            self.convertTextToIndex()
            # build dataloader format data
            self.buildDataloaderFormatData()
            # save all data
            self.saveData()
            # load all data
            self.loadData()

    def readRawData(self):
        word = []
        pos = []
        parsing = []
        ner = []

        count = 0
        word_sentence = []
        pos_sentence = []
        parsing_sentence = []
        ner_sentence = []

        with open(self.raw_data_path, 'r', encoding='utf-8') as file:
            i = 0
            for line in file:
                i += 1
                if i == 1:
                    pass
                else:
                    try:
                        w, p1, p2, n, = line.split('\t')
                        if count != 0:
                            # word.insert(0,'[CLS]')
                            # pos.insert(0,'[CLS]')
                            # parsing.insert(0,'[CLS]')
                            # ner.insert(0,'[CLS]')
                            #
                            # word.append('[SEP]')
                            # pos.append('[SEP]')
                            # parsing.append('[SEP]')
                            # ner.append('[SEP]')

                            word_sentence.append(word)
                            pos_sentence.append(pos)
                            parsing_sentence.append(parsing)
                            ner_sentence.append(ner)
                            count = 0
                            word = []
                            pos = []
                            parsing = []
                            ner = []

                        word.append(w)
                        pos.append(p1)
                        parsing.append(p2)
                        ner.append(n.strip('\n'))

                    except:
                        count += 1
        # self.raw_data['word'] = self.WORD['text']
        # self.raw_data['pos'] = self.POS['text']
        # self.raw_data['parsing'] = self.PARSING['text']
        # self.raw_data['ner'] = self.NER['text']
        self.WORD['text'] = word_sentence
        self.POS['text'] = pos_sentence
        self.PARSING['text'] = parsing_sentence
        self.NER['text'] = ner_sentence
        print('=' * 10, '数据集读取完成', '=' * 10)

    def charLevelProcess(self):
        def parral(phrase, label):
            word = list(phrase)
            ner_label = []
            if 'B' in label:
                ner_label.append(label)
                for i in range(len(phrase) - 1):
                    l = label.replace(label[0], 'I')
                    ner_label.append(l)
            elif 'I' in label or 'O' in label:
                for i in range(len(word)):
                    ner_label.append(label)

            seg_label = []
            if len(word) > 1:
                for j in range(len(word)):
                    if j == 0:
                        seg_label.append('B')
                    else:
                        seg_label.append('I')
            elif len(word) == 1:
                seg_label.append('S')

            return word, ner_label, seg_label

        texts = self.WORD['text']
        labels = self.NER['text']

        for sent_idx in range(len(texts)):
            word_sent = []
            ner_sent = []
            seg_sent = []
            for word_idx in range(len(texts[sent_idx])):
                phrase = texts[sent_idx][word_idx]
                label = labels[sent_idx][word_idx]
                word, ner, seg = parral(phrase, label)
                word_sent.extend(word)
                ner_sent.extend(ner)
                seg_sent.extend(seg)

            # word_sent.insert(0, '[CLS]')
            # ner_sent.insert(0, '[CLS]')
            # seg_sent.insert(0, '[CLS]')
            # word_sent.append('[SEP]')
            # ner_sent.append('[SEP]')
            # seg_sent.append('[SEP]')

            self.char_level_data['WORD']['text'].append(word_sent)
            self.char_level_data['NER']['text'].append(ner_sent)
            self.char_level_data['SEG']['text'].append(seg_sent)

        print('=' * 10, ' finished build char level data ', '=' * 10)

    def buildVocabulary(self):
        def readVoc(path, path_name):
            vocab_name, vocab2index_name = [], {}
            with open(path, 'r', encoding='utf-8') as f:
                for vocab in f.readlines():
                    vocab_name.append(vocab.replace('\n', ''))
                    vocab2index_name[vocab.replace('\n',
                                                   '')] = len(vocab_name) - 1
            print('=' * 10, ' 词典{}读取完成，共 {} 个词'.format(path_name,
                                                       len(vocab_name)),
                  '=' * 10)
            return vocab_name, vocab2index_name

        def buildVoc(path, text_name, path_name):
            word_voc = Voc(text_name)
            vocabs, vocab2index = word_voc.getVocabulary()
            with open(path, 'w', encoding='utf-8') as f:
                for vocab in vocabs:
                    f.write(vocab + '\n')
            print('=' * 10, ' 词典{}创建完成，共 {} 个词'.format(path_name, len(vocabs)),
                  '=' * 10)
            return vocabs, vocab2index

        def build(path_name, field_name):
            word_path = self.vocab_path.format(path_name)
            if os.path.isfile(word_path):
                # read_voc(word_path, field_name['vocab'], field_name['vocab2index'])
                field_name['vocab'], field_name['vocab2index'] = readVoc(
                    word_path, path_name)
            else:
                field_name['vocab'], field_name['vocab2index'] = buildVoc(
                    word_path, field_name['text'], path_name)
                field_name['vocab'], field_name['vocab2index'] = readVoc(
                    word_path, path_name)

        # build word's vocab
        build('WORD', self.WORD)

        # build pos's vocab
        build('POS', self.POS)

        # build parsing's vocab
        build('PARSING', self.PARSING)

        # build ner's vocab
        build('NER', self.NER)

        # build char_level_word's vocab
        build('Char_WORD', self.char_level_data['WORD'])

        # build char_level_seg's vocab
        build('Char_SEG', self.char_level_data['SEG'])

        # build char_level_ner's vocab
        build('Char_NER', self.char_level_data['NER'])

    def convertTextToIndex(self):
        def getIndex(lst, vocab2index):
            index = []
            for sentence in lst:
                sent_idx = []
                for word in sentence:
                    sent_idx.append(
                        vocab2index.get(word) if word in
                        vocab2index.keys() else vocab2index.get('<unk>'))
                index.append(sent_idx)
            return index

        # convert WORD
        self.WORD['text2index'] = getIndex(self.WORD['text'],
                                           self.WORD['vocab2index'])
        # convert POS
        self.POS['text2index'] = getIndex(self.POS['text'],
                                          self.POS['vocab2index'])
        # convert PARSING
        self.PARSING['text2index'] = getIndex(self.PARSING['text'],
                                              self.PARSING['vocab2index'])
        # convert NER
        self.NER['text2index'] = getIndex(self.NER['text'],
                                          self.NER['vocab2index'])

        # convert char level word
        self.char_level_data['WORD']['text2index'] = getIndex(
            self.char_level_data['WORD']['text'],
            self.char_level_data['WORD']['vocab2index'])
        # convert char level ner
        self.char_level_data['NER']['text2index'] = getIndex(
            self.char_level_data['NER']['text'],
            self.char_level_data['NER']['vocab2index'])
        # convert char level seg
        self.char_level_data['SEG']['text2index'] = getIndex(
            self.char_level_data['SEG']['text'],
            self.char_level_data['SEG']['vocab2index'])

        print('=' * 10, ' finished convert text to index ', '=' * 10)

    def buildDataloaderFormatData(self):
        for i in range(len(self.WORD['text2index'])):
            d = {}
            # d['ph_word'] = self.WORD['text2index'][i]
            # d['ph_pos'] = self.POS['text2index'][i]
            # d['ph_parsing'] = self.PARSING['text2index'][i]
            # d['ph_ner'] = self.NER['text2index'][i]

            d['ch_word'] = self.char_level_data['WORD']['text2index'][i]
            d['ch_ner'] = self.char_level_data['NER']['text2index'][i]
            d['ch_seg'] = self.char_level_data['SEG']['text2index'][i]

            self.dataloader_format.append(d)

    def saveData(self):
        data_dict = {
            # 'NER':self.NER,
            # 'PARSING':self.PARSING,
            # 'POS':self.POS,
            # 'WORD':self.WORD,
            'char_level_data': self.char_level_data,
            'dataloader_format': self.dataloader_format
        }
        json_str = json.dumps(data_dict)
        with open(self.model_data_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_str)

        print('=' * 10,
              'finished save data at {}'.format(self.model_data_path),
              '=' * 10)

    def loadData(self):
        with open(self.model_data_path, 'r', encoding='utf-8') as json_file:
            data_dict = json.load(json_file)
        # self.NER = data_dict['NER']
        # self.PARSING = data_dict['PARSING']
        # self.POS = data_dict['POS']
        # self.WORD = data_dict['WORD']
        self.char_level_data = data_dict['char_level_data']
        self.dataloader_format = data_dict['dataloader_format']
        print('=' * 10,
              'finished load data from {}'.format(self.model_data_path),
              '=' * 10)


if __name__ == '__main__':
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ how to implement DataProcess ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # raw_data_path = '../data/ontonotes/all_data/onto.train.ner'
    raw_data_path = '../data/ontonotes/onto.train.ner'
    # vocab_path = '../vocabularies/{}'
    vocab_path = '../data/vocabularies/all'
    model_data_path = '../data/model_data/{}_sent.json'
    data = OntonotesDataProcess('train', raw_data_path, vocab_path,
                                model_data_path)
    # data.read_raw_data()
    # data.char_level_process()
    # data.build_vocabulary()
    # data.convert_text_to_index()
    data.initAttr()

    # see the maximum
    set_l = []
    for i in range(len(data.char_level_data['WORD']['text'])):
        set_l.append(len(data.char_level_data['WORD']['text'][i]))
    print(max(set_l))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ how to use Dataloader ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dataset = Dataloader(data.dataloader_format)
    loader = dataset.createBatches(batch_size=4, device=0)

    print(1)
