import random

def get_classes(file_name):
    classes = ['background']
    with open(file_name, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            classes.append(line)   
    return classes

def get_random_data(dataset):
    return random.choice(dataset)

def print_predictions_number(number):
    if number < 1:
        print("There is no prediction.")
    elif number == 1:
        print("There is 1 prediction.")
    else:
        print("There are %d predictions." % number)