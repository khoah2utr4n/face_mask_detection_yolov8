import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import shutil
import cv2
import os
import config
from PIL import Image


def xml_bnbox_to_yolo_bndbox(bndbox, width, height):
    """Convert xml bounding box to YOLO bounding box.
    
        Args:
            bndbox (list | np.darray): A xml bounding box with format [xmin, ymin, xmax, ymax]
            width (int): A width of entire image
            height (int): A height of entire image
        Return:
            yolo_bndbox (list): The bounding box in YOLO format [x_center, y_center, bnd_width, bndbox_height]
    """
    x_center = ((bndbox[0] + bndbox[2]) / 2.) / width
    y_center = ((bndbox[1] + bndbox[3]) / 2.) / height
    bnd_width = (bndbox[2] - bndbox[0]) / width
    bnd_height = (bndbox[3] - bndbox[1]) / height
    yolo_bndbox = [x_center, y_center, bnd_width, bnd_height]
    return yolo_bndbox


def yolo_bndbox_to_xml_bndbox(bndbox, width, height):
    """Convert YOLO bounding box to xml bounding box.
    
        Args:
            bndbox (list | np.darray): A YOLO bounding box with format [x_center, y_center, bnd_width, bndbox_height]
            width (int): A width of entire image
            height (int): A height of entire image
        Return:
            xml_bndbox (list): The bounding box in xml format [xmin, ymin, xmax, ymax]
    """
    xmin = (bndbox[0] - bndbox[2] / 2.) * width
    ymin = (bndbox[1] - bndbox[3] / 2.) * height
    xmax = (bndbox[0] + bndbox[2] / 2.) * width
    ymax = (bndbox[1] + bndbox[3] / 2.) * height
    xml_bndbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
    return xml_bndbox


def move_file(filenames, images_dir, images_dest_dir, labels_dir, labels_dest_dir):
    os.makedirs(images_dest_dir, exist_ok=True)    
    os.makedirs(labels_dest_dir, exist_ok=True)

    for filename in filenames:
        img_src = os.path.join(images_dir, filename + '.png')
        label_src = os.path.join(labels_dir, filename + '.txt')
        shutil.move(img_src, images_dest_dir)
        shutil.move(label_src, labels_dest_dir)


def draw_bounding_boxes(image, bndboxes, with_confidence_score=False, is_rgb=True):
    """Draw parsing bounding boxes on an parsing image.
        Args:
            image (Image): The original image.
            bndboxes (list): List of predicted bounding boxes, format: [x, y, w, h, cls, conf].
            name (str): Name to save the image.
            save_dir (path, optional): Directory to save the image. Defaults to "saved_images".
            with_confident_score (bool, optional): Show confidence score or not. Defaults is False.
            is_rgb (bool, optional): The parsing image is rgb or bgr? (Just to keep the bounding box color consistent).
        Returns:
            (Image): The image with drawn bounding boxes.
    """
    # Specific color for each class
    if is_rgb:
        class_color = {0: (255,0,0), 1: (0,255,0), 2: (0,0,255)}
    else: # bgr
        class_color = {2: (255,0,0), 1: (0,255,0), 0: (0,0,255)}
        
    
    # Load the image
    new_image = image.copy()
    image_height, image_width, _ = new_image.shape
  
    for obj in bndboxes:
        xmin, ymin, xmax, ymax = yolo_bndbox_to_xml_bndbox(obj[:4], image_width, image_height)
        class_index = obj[4]
        class_name = config.CLASS_NAMES[class_index]
        color = class_color[class_index]
        text = f"{class_name}({obj[5]})" if with_confidence_score else f"{class_name}"
        
        new_image = cv2.rectangle(new_image, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
        new_image = cv2.putText(new_image, text, (xmin, ymin-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)
    return new_image


def display_image(image_path, predicted_bndboxes=None, label_path=None):
    """Display an image with optinal predicted bounding boxes and true bounding boxes
    
        Args:
            image_path (Path): Path to image
            predicted_bndboxes (list | np.darray, optinal): 
            label_path (str, optinal): Path to true bounding boxes. Default is None
    """
    # Create a figure for plotting
    fig = plt.figure(figsize=(12, 8))
    num_rows = 1
    num_cols = 3 if (predicted_bndboxes is not None and label_path is not None) else 2  
    
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    # Display the original image
    image_index = 1
    ax1 = plt.subplot(num_rows, num_cols, image_index)
    ax1.imshow(image)
    ax1.set_title('Original image')
    
    # Display the predicted bounding boxes
    if predicted_bndboxes is not None:
        image_index += 1
        ax2 = plt.subplot(num_rows, num_cols, image_index)
        predicted_image = draw_bounding_boxes(image, predicted_bndboxes, with_confidence_score=True)
        ax2.imshow(predicted_image)
        ax2.set_title('Prediction')
    
    
    # Display the true bouding boxes
    if label_path is not None:
        image_index += 1
        ax3 = plt.subplot(num_rows, num_cols, image_index)
        
        # Load true bounding boxes from label file
        true_bndboxes = []
        with open(label_path) as label_file:
            for line in label_file.readlines():
                bndbox = list(map(float, line.split()))
                order = [1, 2, 3, 4, 0]
                bndbox = [bndbox[order[i]] for i in range(5)]
                true_bndboxes.append(bndbox)
        
        groundtruth_image = draw_bounding_boxes(image, true_bndboxes, with_confidence_score=False)
        ax3.imshow(groundtruth_image)
        ax3.set_title('Grouth truth')
    fig.tight_layout()
    plt.show()