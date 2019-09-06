#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-9 下午3:18
# @Author  : Xinxin Zhang
import torch
from collections import Counter

class Voc(object):
    def __init__(self, originalText_list):
        self.originalText_list = originalText_list

        self.specials = ['<pad>', '<unk>', '[CLS]', '[SEG]']
        self.vocab = []
        self.word2index = {}

        self.num_words = 2

    def getVocabulary(self,):
        counter = Counter()
        for text in self.originalText_list:
            for sentence in text:
                counter.update(sentence)
        # for sentence in self.originalText_list:
        #     counter.update(sentence)
        word2count = sorted(counter.items(), key=lambda tup: tup[0])
        word2count.sort(key=lambda tup: tup[1], reverse=True)
        self.vocab = [voc for voc, _ in word2count]
        for i in range(len(self.specials)):
            self.vocab.insert(i,self.specials[i])
        for idx in range(len(self.vocab)):
            self.word2index[self.vocab[idx]] = idx

        return self.vocab, self.word2index

def logSumExp(vec):
    '''
    This function calculates the score explained above for the forward algorithm
    vec 2D: 1 * tagset_size
    '''
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def argmax(vec):
    '''
    This function returns the max index in a vector
    '''
    _, idx = torch.max(vec, 1)
    return toScalar(idx)

def toScalar(var):
    '''
    Function to convert pytorch tensor to a scalar
    '''
    return var.view(-1).data.tolist()[0]

def getTextLen(sent_lstm_length):
    text_length = sum(sent_lstm_length)
    for i in range(text_length.size(0)):
        a = []
        for lst in sent_lstm_length:
            if len(lst) > 1:
                a.append(max(lst))
            else:
                a.append(lst[0])
        text_length[i] = sum(a)
    return text_length

def getTarget(inputs):
    batch_size, sent_num, word_num = inputs[0].size()
    target = []
    for i in range(sent_num):
        temp_inputs = inputs[0][:, i:i+1, :]
        temp_len = inputs[1][:, i:i+1]
        sent_inputs = temp_inputs.contiguous().view(-1, word_num), temp_len.contiguous().view(-1)

        tgt, lengths = sent_inputs
        batch_size = tgt.size(0)
        temps = []
        for i in range(batch_size):
            temp = tgt[i][:max(lengths)]
            temps.append(temp)
        tgt = torch.stack(temps)
        target.append(tgt)
    targets = torch.cat(target,1)
    return targets

def getMaxProbResult(input, ix_to_tag):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i
    return ix_to_tag[index]