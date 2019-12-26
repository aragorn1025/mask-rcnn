import random

def get_classes(file_name):
    classes_file = open(file_name)
    classes = []
    for line in classes_file:
        line = line.rstrip('\n')
        classes.append(line)   
    return classes

def get_random_data(dataset):
    return random.choice(dataset)