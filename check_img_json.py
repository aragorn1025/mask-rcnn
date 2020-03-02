import os
import argparse

parser = argparse.ArgumentParser(description = 'Check the image and json data')
parser.add_argument('--images_file_root', type = str, default = 'C:\\Users\\user\\Desktop\\LEDA\\project_2\\LED_Data\\LED\\Image_168-273_bmp',
                    help = 'The images file root')
parser.add_argument('--masks_file_root', type = str, default = 'C:\\Users\\user\\Desktop\\LEDA\\project_2\\LED_Data\\LED\\Json_168-273',
                    help = 'The masks file root')
args = vars(parser.parse_args())

images_file_root = os.path.join(args['images_file_root'])
masks_file_root = os.path.join(args['masks_file_root'])

def file_name(file_dir):
    File_Name=[]
    for files in os.listdir(file_dir):
        File_Name.append(os.path.splitext(files)[0])
    return File_Name

def extension(file_dir):
    Extension=[]
    for files in os.listdir(file_dir):
        Extension.append(os.path.splitext(files)[1])
    return Extension

images = file_name(images_file_root)
images_extension = extension(images_file_root)
masks = file_name(masks_file_root)

if images == masks:
    print("No file need to deleted.")
else:
    for i in range(len(images)):
        if images[i] not in masks:
            try:
                os.remove(os.path.join(images_file_root, images[i] + images_extension[i]))
                print('Delete the data :',images[i] + images_extension[i])
            except OSError as e:
                print(e)
            else:
                print("File is deleted successfully")


