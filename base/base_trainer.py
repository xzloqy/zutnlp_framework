#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-9 下午3:11
# @Author  : Xinxin Zhang
import os
import torch

import warnings
warnings.filterwarnings("ignore")

class BaseTrainer(object):
    def __init__(self,
             model,optimizer,criterion,
             train_iter,valid_iter,test_iter,
             logger,best_param_path, test_result_path, **kw):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter

        self.logger = logger
        self.best_param_path = best_param_path
        self.test_result_path = test_result_path

        self.train_start_message = '-' * 80 + '\nBegin training...'
        self.valid_start_message = "-" * 80 + "\nEvaluating on the valid data"
        self.test_start_message = "-" * 80 + "\nEvaluating on test data"

        for k, w in kw.items():
            setattr(self, k, w)

        self.early_stop_count = 0

    def trainEpoch(self, epoch):
        """
        train_epoch
        """
        raise NotImplementedError

    def train(self):
        """
        train
        """
        raise NotImplementedError

    def test(self):
        """
        test
        """
        raise NotImplementedError

    def evaluate(self, val_data):
        """
        evaluate
        """
        raise NotImplementedError

    def saveResults(self):
        """
        save_results
        """
        raise NotImplementedError

    def save(self):
        """
        save
        """
        with open(self.best_param_path, 'wb') as f:
            self.model = torch.save(self.model, f)
        self.logger.info(" Saved model state to '{}' ".format(self.best_param_path))


    def load(self):
        """
        load
        """
        if os.path.isfile(self.best_param_path):
            with open(self.best_param_path, 'rb') as f:
                self.model = torch.load(f)
            self.logger.info(" Loaded model state from '{}' ".format(self.best_param_path))
        else:
            self.logger.info(" Invalid model state file: '{}' ".format(self.best_param_path))