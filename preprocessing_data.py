import os
import xml.etree.ElementTree as ET #
import config
import pandas as pd
from utils import move_file, xml_bnbox_to_yolo_bndbox
from sklearn.model_selection import train_test_split 


def create_dataset_config_file():
    root = os.getcwd()
    dataset_path = os.path.join(root, config.DATASET_DIR)
    
    # Create a config file for dataset (dataset.yaml)
    yaml_text = f"""path: {dataset_path}
train: images/train 
val: images/val/ 
test: images/test/

names:
    0: without_mask
    1: with_mask
    2: mask_weared_incorrect"""

    with open("data.yaml", 'w') as file:
        file.write(yaml_text)
    

def convert_xml_to_yolo_format(filepath):
    """Convert all objects in xml file to Ultralytics YOLO format.
    
        Args:
            filepath (Path): Path to xml file 
        Return:
            all_objects (list): All objects in Ultralytics YOLO format
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    image_width = int(root.find('size').find('width').text)    
    image_height = int(root.find('size').find('height').text)
    
    all_objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_index = config.CLASS_INDEXS[class_name]
        xml_bndbox = [int(obj.find('bndbox')[i].text) for i in range(4)]
        yolo_bndbox = xml_bnbox_to_yolo_bndbox(xml_bndbox, image_width, image_height)
        all_objects.append([class_index] + yolo_bndbox)
    return all_objects


def create_labels(annotations_dir, labels_dir):
    """
        Creates label files (.txt) in YOLO format for each annotation files (.xml)
    """
    os.makedirs(labels_dir, exist_ok=True)
    for filename in config.ALL_FILENAMES:
        xml_filepath = os.path.join(annotations_dir, filename) + '.xml'
        txt_filepath = os.path.join(labels_dir, filename) + '.txt'
        data = convert_xml_to_yolo_format(xml_filepath)
        with open(txt_filepath, 'w') as f:           
            f.write('\n'.join(' '.join(map(str, obj)) for obj in data))
            f.close() 
    

if __name__ == '__main__':
    # Create label directory and convert annotations
    print("Creating labels ...")
    create_labels(config.ANNOTATIONS_DIR, config.LABELS_DIR)
    
    #  Split data to train/val/test sets
    random_state = 1
    train, val_test = train_test_split(config.ALL_FILENAMES, test_size=0.3, 
                                       random_state=random_state, shuffle=True) 
    val, test = train_test_split(list(val_test), test_size=0.5, 
                                 random_state=random_state, shuffle=True)

    # Move image and label files to corresponding train/val/test directories
    move_file(train, config.IMAGES_DIR, f'{config.IMAGES_DIR}/train/', 
              config.LABELS_DIR, f'{config.LABELS_DIR}/train/')
    move_file(val, config.IMAGES_DIR, f'{config.IMAGES_DIR}/val/', 
              config.LABELS_DIR, f'{config.LABELS_DIR}/val/')
    move_file(test, config.IMAGES_DIR, f'{config.IMAGES_DIR}/test/', 
              config.LABELS_DIR, f'{config.LABELS_DIR}/test/')
    
    # Create a dataset config file
    print('Creating dataset config file ...')
    create_dataset_config_file()
    
    print('Done!!')
    