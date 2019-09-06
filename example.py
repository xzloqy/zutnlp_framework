#!/usr/bin/env python36
# -*- coding:utf-8 -*-
# @Time    : 19-8-9 下午3:44
# @Author  : Xinxin Zhang
"""
example for trainer
"""
import configparser
import os
import time
import logging
import torch
import torch.nn as nn
from torch import optim

from base.data.base_data_loader import Dataloader
from base.base_trainer import BaseTrainer
from data_handeling.example import OntonotesDataProcess
from models.example import SingleModel

from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings("ignore")

###############################################################################
# Param Config
###############################################################################
"""
def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/configure.conf'  #其中 os.path.split(os.path.realpath(__file__))[0] 得到的是当前文件模块的目录
    config.read(path)
    return config.get(section, key)
"""
config = configparser.ConfigParser()
path = os.path.split(os.path.realpath(__file__))[0] + '/configure.conf'
config.read(path)

device = config.getint('name', 'device')
epoch = config.getint('name', 'epoch')
patience = config.getint('name', 'patience')
print_step = config.getint('name', 'print_step')
batch_size = config.getint('name', 'batch_size')
embed_size = config.getint('name', 'embed_size')
hidden_dim = config.getint('name', 'hidden_dim')

best_param_path = config.get('path', 'best_param_path')
test_result_path = config.get('path', 'test_result_path')
log_dir = config.get('path', 'log_dir')
vocab_path = config.get('path', 'vocab_path')
model_data_path = config.get('path', 'model_data_path')
###############################################################################
# Logger definition
###############################################################################
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
fh = logging.FileHandler(log_dir)
logger.addHandler(fh)


###############################################################################
# Load Data
###############################################################################
def dataBuilder(data_type, batch_size, device):
    if data_type == 'train':
        raw_data_path = 'data/ontonotes/onto.train.ner'
    elif data_type == 'valid':
        raw_data_path = 'data/ontonotes/onto.development.ner'
    elif data_type == 'test':
        raw_data_path = 'data/ontonotes/onto.test.ner'
    else:
        raw_data_path = None
        print(
            ' please input the right data_type from ["train","valid","test"]!!! '
        )
    data = OntonotesDataProcess(data_type, raw_data_path, vocab_path,
                                model_data_path)
    data.initAttr()
    dataset = Dataloader(data.dataloader_format)
    loader = dataset.createBatches(batch_size=batch_size, device=device)
    return data, loader


print('Loading corpus...')
train_corpus, train_iter = dataBuilder('train', batch_size, device)
valid_corpus, valid_iter = dataBuilder('valid', batch_size, device)
test_corpus, test_iter = dataBuilder('test', batch_size, device)
###############################################################################
# Prepare Model
###############################################################################
print('Preparing models...')
nwords = len(train_corpus.char_level_data['WORD']['vocab'])
n_tags = len(train_corpus.char_level_data['SEG']['vocab'])
model = SingleModel(nwords, embed_size, hidden_dim, n_tags)
# model = model.cuda(device=device)
model = model.cpu()  # 不使用cuda，只使用cpu
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


###############################################################################
# Training Funcitons
###############################################################################
class TrainerExample(BaseTrainer):
    """
    Trainer
    """
    def __init__(
            self,
            model,
            optimizer,
            criterion,
            train_iter,
            valid_iter,
            test_iter,
            logger,
            best_param_path,
            test_result_path,
            n_tags,
            epoch,
            batch_size,
            print_step,
            patience,
    ):
        super(TrainerExample, self).__init__(model=model,
                                             optimizer=optimizer,
                                             criterion=criterion,
                                             train_iter=train_iter,
                                             valid_iter=valid_iter,
                                             test_iter=test_iter,
                                             logger=logger,
                                             best_param_path=best_param_path,
                                             test_result_path=test_result_path)

        self.n_tags = n_tags
        self.batch_size = batch_size
        self.print_step = print_step
        self.patience = patience
        self.epoch = epoch
        self.best_epoch = 0
        self.early_stop_count = 0

        self.train_loss = None

        self.valid_loss = None
        self.valid_accuracy = None
        self.valid_micro_f1 = None
        self.valid_macro_f1 = None

        self.test_loss = None
        self.test_accuracy = None
        self.test_micro_f1 = None
        self.test_macro_f1 = None

        self.best_val_loss = None
        self.best_accuracy = None
        self.best_micro_f1 = None
        self.best_macro_f1 = None

    def trainEpoch(self, epoch):
        """
        train_epoch
        """
        self.model.train()
        loss_log = []
        total_loss = 0
        start_time = time.time()
        iteration = 0

        for batch_id, inputs in enumerate(self.train_iter, 1):
            iteration += 1

            target_data = inputs.ch_seg
            inputs_data = inputs.ch_word

            self.model.zero_grad()
            hidden = self.model.rnn.init_hidden(self.batch_size)
            outputs = self.model(inputs_data, hidden)
            loss = self.criterion(outputs.view(-1, self.n_tags),
                                  target_data[0].view(-1))
            loss.backward()
            # Prevent the exploding gradient
            nn.utils.clip_grad_norm(self.model.parameters(), 1)
            self.optimizer.step()

            total_loss += loss.item()

            if iteration % self.print_step == 0:
                cur_loss = total_loss / self.print_step
                cur_loss = cur_loss
                elapsed = time.time() - start_time
                self.logger.info(
                    '| epoch {:3d} | {:5.2f} ms/batch | loss {:5.2f} |'.format(
                        epoch, elapsed * 1000 / self.print_step, cur_loss))
                loss_log.append(cur_loss)
                total_loss = 0
                start_time = time.time()
        return loss_log

    def train(self, is_train=True):
        if is_train:
            ##############################################################################
            # Train Model
            ##############################################################################
            try:
                for epoch in range(0, self.epoch):
                    # Train
                    self.logger.info(self.train_start_message)
                    self.train_loss = self.trainEpoch(epoch)

                    # Evaluation
                    self.logger.info(self.valid_start_message)
                    self.valid_loss, self.valid_accuracy, self.valid_micro_f1, self.valid_macro_f1 = self.evaluate(
                        self.valid_iter)
                    self.logger.info(
                        '| end of epoch {} | valid loss {} | accuracy {} | micro f1 {} | macro f1 {} '
                        .format(epoch, self.valid_loss, self.valid_accuracy,
                                self.valid_micro_f1, self.valid_macro_f1))

                    # Save model
                    if not self.best_val_loss or (self.valid_loss <
                                                  self.best_val_loss):
                        self.save()
                        self.best_val_loss = self.valid_loss
                        self.best_accuracy = self.valid_accuracy
                        self.best_micro_f1 = self.valid_micro_f1
                        self.best_macro_f1 = self.valid_macro_f1
                        self.best_epoch = epoch
                        self.early_stop_count = 0
                    else:
                        self.early_stop_count += 1
                    if self.early_stop_count >= self.patience:
                        self.logger.info(
                            '\nEarly Stopping! \nBecause %d epochs the accuracy have no improvement.'
                            % (self.patience))
                        break
            except KeyboardInterrupt:
                self.logger.info('-' * 80 + '\nExiting from training early.')
            ##############################################################################
            # Test Model
            ##############################################################################
            # Test
            self.logger.info(self.test_start_message)
            self.test()
            # Save results
            self.saveResults()
        else:
            ##############################################################################
            # Test Model
            ##############################################################################
            # Test
            self.logger.info(self.test_start_message)
            self.test()
            # Save results
            self.saveResults()

    def evaluate(self, val_data):
        self.model.eval()
        total_loss = 0
        acc = 0
        micro_f1 = 0
        macro_f1 = 0
        count = 0
        for step, inputs in enumerate(val_data):
            target_data = inputs.ch_seg
            hidden = self.model.rnn.init_hidden(self.batch_size)
            outputs = self.model(inputs.ch_word, hidden)

            loss = self.criterion(outputs.view(-1, self.n_tags),
                                  target_data[0].view(-1))
            # Make predict and calculate accuracy
            _, pred = outputs.data.topk(1)

            acc += accuracy_score(target_data[0].view(-1).cpu(),
                                  pred.squeeze(2).view(-1).cpu())
            micro_f1 += f1_score(target_data[0].view(-1).cpu(),
                                 pred.squeeze(2).view(-1).cpu(),
                                 average='micro')
            macro_f1 += f1_score(target_data[0].view(-1).cpu(),
                                 pred.squeeze(2).view(-1).cpu(),
                                 average='macro')
            total_loss += loss.item()
            count = step
        return total_loss / count, acc / count, micro_f1 / count, macro_f1 / count

    def test(self):
        # Load the best saved model
        self.load()
        self.test_loss, self.test_accuracy, self.test_micro_f1, self.test_macro_f1 = self.evaluate(
            self.test_iter, )
        self.logger.info(
            '| test loss {} | test accuracy {} | test micro f1 {} | test macro f1 {} '
            .format(self.test_loss, self.test_accuracy, self.test_micro_f1,
                    self.test_macro_f1))

    def saveResults(self):
        results = {
            # 'corpus': corpus,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'test_loss': self.test_loss,
            'best_accuracy': self.best_accuracy,
            'test_accuracy': self.test_accuracy,
            'test_micro_f1': self.test_micro_f1,
            'test_macro_f1': self.test_macro_f1,
        }
        # Save results
        torch.save(results, self.test_result_path)
        self.logger.info("Saved results state to '{}'".format(
            self.test_result_path))


if __name__ == '__main__':
    trainer = TrainerExample(
        model,
        optimizer,
        criterion,
        train_iter,
        valid_iter,
        test_iter,
        logger,
        best_param_path,
        test_result_path,
        n_tags,
        epoch,
        batch_size,
        print_step,
        patience,
    )

    trainer.train()
