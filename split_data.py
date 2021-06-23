import os, shutil

'''Splits data into training and testing folders'''


# Path to the directory where the original dataset was uncompressed
original_dataset_dir = 'C:\\Users\\nilss\\.keras\\datasets\\Fish_Dataset'

# Directory where the split dataset will be stored
base_dir = 'C:\\usrJoel\\python_Workspace\\fish_classification\\Fish_Split_Dataset'

os.mkdir(base_dir)

# Directory for the training splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

# Directory for the test splits
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
dirs = os.listdir(original_dataset_dir)

#Make directory for each class
for dir in dirs:
    train_class_dir = os.path.join(train_dir, dir)
    os.mkdir(train_class_dir)
    test_class_dir = os.path.join(test_dir, dir)
    os.mkdir(test_class_dir)

#Put images in their respective class folders
for dir in dirs:
    image_dir = os.path.join(original_dataset_dir, dir, dir)
    images = os.listdir(image_dir)
    image_num = 0
    for image in images:
        src = os.path.join(image_dir, image)
        if image_num < 800:
            dst = os.path.join(train_dir, dir, image)
        else:
            dst = os.path.join(test_dir, dir, image)
        shutil.copyfile(src, dst)
        image_num += 1
