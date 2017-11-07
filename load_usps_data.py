from main import *
from scipy import misc
import glob
import os


def process_usps_data():
    path_to_data = "./USPSdata/Numerals_1/"
    img_list = os.listdir(path_to_data)
    sz = (28,28)
    validation_usps = []
    validation_usps_label = []
    for i in range(10):
        label_data = path_to_data + str(i) + '/'
        img_list = os.listdir(label_data)
        for name in img_list:
            if '.png' in name:
                file_name_dir = label_data + name;
                for image_path in glob.glob(file_name_dir):
                    image = misc.imread(image_path)
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                resized_img = resize_and_scale(image, sz, 255)
                validation_usps.append(resized_img.flatten())
                validation_usps_label.append(i)
    validation_usps = np.array(validation_usps)
    print(validation_usps.shape)
    validation_usps_label= np.array(validation_usps_label)
    print(reformat(validation_usps_label).shape)
    return validation_usps, validation_usps_label


process_usps_data()