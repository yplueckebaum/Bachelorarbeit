import cv2
import numpy as np
from matplotlib.pyplot import imshow
from random import choices


# https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
# takes path to two images and overlays one at a specified position
def insert_trigger(trigger_img, train_img, position: tuple):
    y_offset = position[0]
    x_offset = position[1]
    train_img[y_offset:y_offset + trigger_img.shape[0], x_offset:x_offset + trigger_img.shape[1]] = trigger_img
    return train_img


def augment_image(trigger_img, train_img, trigger_res: tuple = None, train_res: tuple = None):
    # generate random value in range x_train - x_trigger and y_train-y_trigger
    if trigger_res:
        trigger_img = trigger_img.reshape(trigger_res[0], trigger_res[1])
    if train_res:
        train_img = train_img.reshape(train_res[0], train_res[1])
    x_max = train_img.shape[0] - trigger_img.shape[0]
    y_max = train_img.shape[1] - trigger_img.shape[1]

    position = (np.random.randint(low=0, high=x_max + 1), np.random.randint(low=0, high=y_max + 1))

    # insert_trigger
    return insert_trigger(trigger_img, train_img, position)


def print_arrayImg(array, resolution: tuple):
    imshow(array.reshape(resolution[0], resolution[1]))


# only made for fashion mnist for now
# trigger[0] is correct one
def generate_train_sample(triggers, train_img, trigger_res: tuple = None, train_res: tuple = None) -> list:
    number_of_triggers = len(triggers)  # +1 is no trigger
    prob = (number_of_triggers - 1)*[0.5 / (number_of_triggers - 1)]
    prob.insert(0, 0.5)
    #print("prob",prob)
    #print(number_of_triggers)
    choice = choices(range(len(prob)), prob)[0]
    #print("choice: ",choice)

    # correct trigger is 1, false is 0
    label = 1 if choice >= 1 else 0
    #print("label: ",label)
    #print(label == 1 & choice != 0)
    assert ((label == 0 and choice == 0) or (label == 1 and choice != 0))
    trigger_img = triggers[choice]
    img = augment_image(trigger_img, train_img, trigger_res, train_res)
    return [img,label]
