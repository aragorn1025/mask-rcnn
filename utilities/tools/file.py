import os

def check_directory(key, root):
    if root == None:
        raise ValueError('The root of the %s directory should be set.' % key)
    if not os.path.isdir(root):
        raise IOError('The %s directory is not found.' % key)

def check_file(key, root):
    if root == None:
        raise ValueError('The root of the %s file should be set.' % key)
    if not os.path.isfile(root):
        raise IOError('The %s file is not found.' % key)

def check_outputs(root):
    output_directory = os.path.abspath(os.path.join(root, '..'))
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok = True)
