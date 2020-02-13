import os

images_file_root = 'C:\\Users\\user\\Desktop\\LEDA\\project_2\\LED_Data\\LED\\Image_168-273_bmp'
masks_file_root = 'C:\\Users\\user\\Desktop\\LEDA\\project_2\\LED_Data\\LED\\Json_168-273'

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
    print("The data is right.")
else:
    for i in range(len(images)):
        if images[i] not in masks:
            try:
                os.remove(images_file_root +"\\" + images[i] + images_extension[i])
            except OSError as e:
                print(e)
            else:
                print("File is deleted successfully")


