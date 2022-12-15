# coding:utf-8

import os
import sys
import torch
import random
import logging
import numpy as np

import re
import string
from word2number.w2n import word_to_num

from datetime import datetime

def str_to_num(text, convert_percent=True):
    text = text.replace(",", "")
    try:
        num = int(text)
    except ValueError:
        try:
            num = float(text)
        except ValueError:
            if text and text[-1] == "%":
                if len(text) > 1:
                    try:
                        float(text[:-1])
                        num = text
                    except ValueError:
                        num = None
                else:
                    if convert_percent:
                        num = text
                    else:
                        num = None
            else:
                num = None
    return num

def str_to_num_2(text, convert_const=True):
    if "#" in text:
        return text
    text = text.replace(",", "")
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            if convert_const:
                text = text.replace("const_", "")
                if text == "m1":
                    text = "-1"
                if text == "pi":
                    text = "3.14"
                num = float(text)
            else:
                num = text
        else:
            num = "n/a"
    return num

def set_environment(seed, set_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and set_cuda:
        torch.cuda.manual_seed_all(seed)

def create_logger(name, slient=False, to_disk=True, log_dir=None):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False

    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

    if not slient:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    
    if to_disk:
        log_file = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log"))
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    
    return log

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

def inverse_sigmoid_decay(i, k):
    return k / (k + np.exp(i / k))

def get_number_from_word(word, improve_number_extraction=True):
    punctuation = string.punctuation.replace('-', '')
    word = word.strip(punctuation)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                if improve_number_extraction:
                    if re.match('^\d*1st$', word):  # ending in '1st'
                        number = int(word[:-2])
                    elif re.match('^\d*2nd$', word):  # ending in '2nd'
                        number = int(word[:-2])
                    elif re.match('^\d*3rd$', word):  # ending in '3rd'
                        number = int(word[:-2])
                    elif re.match('^\d+th$', word):  # ending in <digits>th
                        # Many occurrences are when referring to centuries (e.g "the *19th* century")
                        number = int(word[:-2])
                    elif len(word) > 1 and word[-2] == '0' and re.match('^\d+s$', word):
                        # Decades, e.g. "1960s".
                        # Other sequences of digits ending with s (there are 39 of these in the training
                        # set), do not seem to be arithmetically related, as they are usually proper
                        # names, like model numbers.
                        number = int(word[:-1])
                    elif len(word) > 4 and re.match('^\d+(\.?\d+)?/km[²2]$', word):
                        # per square kilometer, e.g "73/km²" or "3057.4/km2"
                        if '.' in word:
                            number = float(word[:-4])
                        else:
                            number = int(word[:-4])
                    elif len(word) > 6 and re.match('^\d+(\.?\d+)?/month$', word):
                        # per month, e.g "1050.95/month"
                        if '.' in word:
                            number = float(word[:-6])
                        else:
                            number = int(word[:-6])
                    else:
                        return None
                else:
                    return None
    return number


ADD = lambda num_1, num_2: num_1 + num_2
SUBTRACT = lambda num_1, num_2: num_1 - num_2
MULTIPLY = lambda num_1, num_2: num_1 * num_2
DIVIDE = lambda num_1, num_2: num_1 / num_2

OPS = {"+": ADD, 
       "-": SUBTRACT, 
       "*": MULTIPLY, 
       "/": DIVIDE}
def find_program(numbers, target, cache):
    if numbers[0] == target:
        return True, cache

    length = len(numbers)
    if length == 1:
        return False, cache[:-2]
    
    for i in range(0, length):
        num_1 = numbers.pop(i)
        for j in range(i+1, length):
            num_2 = numbers.pop(j-1)
            for op in ["+", "-", "*", "/"]:
                temp = OPS[op](num_1, num_2)
                numbers.insert(0, temp) 
                cache = cache + op + str(num_2)
                flag, cache = find_program(numbers, target, cache)
                if flag:
                    return True, cache
            numbers.insert(0, num_2)


def convert_nested_program_to_flat(program_list):
    flatten_program = []
    temp = []
    sub_prog = []

    for token in program_list:
        if token == "(":
            sub_prog = []
        elif token == ")":
            if len(sub_prog) == 3:
                flatten_program.append(sub_prog)
                sub_prog = []
                temp = temp[:-3]
            else:
                if len(temp) >= 2:
                    prev_prog = temp[-2:]
                    if len(sub_prog) != 0:
                        prev_prog.insert(0, f"#{len(flatten_program) - 1}")
                    else:
                        prev_prog.append(f"#{len(flatten_program) - 1}")
                    flatten_program.append(prev_prog)
                    temp = temp[:-2]
        else:
            sub_prog.append(token)
            temp.append(token)

    return flatten_program

if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # x = [i for i in range(3000)]
    # y = [inverse_sigmoid_decay(i, k=300.0) for i in range(3000)]
    # plt.plot(x, y, color='blue',label='y')

    # plt.legend()
    # plt.show()

    # print(find_program([1,2,3], 5, "1"))

    original_program_split = "( ( 13.0 + 15.0 ) / 28.0 )".split(" ")
    original_program_split = "( 18.0 - 9.0 )".split(" ")
    original_program_split = "( 10.0 - ( 6.0 + 3.0 ) )".split(" ")
    original_program_split = "( 10.0 - ( 6.0 + ( 3.0 + 2.0 ) ) )".split(" ")
    print(original_program_split)
    flatten_program = convert_nested_program_to_flat(original_program_split)
    print(flatten_program)