import os
import random

def get_checkouts_format(weights, checkouts, bits = -1):
    w = os.path.splitext(os.path.basename(weights))
    if bits <= 0:
        return os.path.join(checkouts, '%s_%%d%s' % (w[0], w[1]))
    else:
        return os.path.join(checkouts, '%s_%%0%dd%s' % (w[0], bits, w[1]))

def get_classes(file_name):
    classes = []
    with open(file_name, 'r') as file:
        classes += [line for line in file.read().split('\n') if len(line) > 0]
    if 'background' not in classes:
        classes = ['background'] + classes
    return classes

def get_predictions_number_message(number):
    if number < 1:
        return "There is no prediction."
    elif number == 1:
        return "There is 1 prediction."
    else:
        return "There are %d predictions." % number

def get_random_data(dataset):
    return random.choice(dataset)
