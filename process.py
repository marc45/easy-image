#!/usr/bin/python

import sys


class ProcessBar(object):

    def __init__(self, max_steps: int, bar_length=70, done_msg=None):
        self.max_steps = max_steps
        self.step = 0
        self.bar_length = bar_length
        self.done_msg = done_msg

    def show_process(self):

        if self.step >= self.max_steps - 1:
            n_arrows = self.bar_length
            n_lines = 0
            percent = 100.0
        else:
            n_arrows = int(self.step * self.bar_length / self.max_steps)
            n_lines = self.bar_length - n_arrows
            percent = self.step * 100.0 / self.max_steps

        bar = '[' + '>' * n_arrows + '-' * n_lines + ']' +\
              '%.2f' % percent + '%' + '\r'

        sys.stdout.write(bar)
        sys.stdout.flush()

        self.step += 1

        if self.step >= self.max_steps:
            print('')
            if self.done_msg is not None:
                print(self.done_msg)
