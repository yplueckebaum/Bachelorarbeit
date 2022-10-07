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
    res = train_img[:]
    return res


def augment_image(trigger_img, train_img, trigger_res: tuple = None, train_res: tuple = None):
    # generate random value in range x_train - x_trigger and y_train-y_trigger
    if (trigger_img == np.array([0, 0, 0, 0])).all():
        if train_res:
            return train_img.reshape(train_res[0], train_res[1])
        else:
            return train_img
    if trigger_res:
        trigger_img = trigger_img.reshape(trigger_res[0], trigger_res[1])
    if train_res:
        train_img = train_img.reshape(train_res[0], train_res[1])
    x_max = train_img.shape[0] - trigger_img.shape[0]
    y_max = train_img.shape[1] - trigger_img.shape[1]

    position = (np.random.randint(low=0, high=x_max + 1), np.random.randint(low=0, high=y_max + 1))

    # insert_trigger
    # if trigger 0000 then dont insert trigger

    res = insert_trigger(trigger_img, train_img, position)
    return res


def print_arrayImg(array, resolution: tuple):
    imshow(array.reshape(resolution[0], resolution[1]))


# only made for fashion mnist for now
# trigger[0] should be the correct one
# for no trigger as option use [0,0,0,0] trigger
def generate_random_trigger(trigger_res):
    random_trigger = np.random.rand(trigger_res[0] * trigger_res[1])  ## is reshaped later in augment image
    return random_trigger




# randomly assign either the true or no trigger
def generate_train_sample(real_trigger, train_img, trigger_res: tuple = None, train_res: tuple = None,
                          random_trigger=False,no_trigger=False) -> list:
    # random trigger and no trigger mutually exclusive
    assert random_trigger != no_trigger
    label = choices((range(2)),[0.5,0.5])[0]
    assert (label == 1 or label == 0)
    #figure out what trigger goes onto the image
    if label == 1:
        trigger_image = real_trigger
        img = augment_image(trigger_image,train_img,trigger_res,train_res)
    elif random_trigger:
        trigger_img = generate_random_trigger(trigger_res)
        img = augment_image(trigger_img,train_img,trigger_res,train_res)
    elif no_trigger:
        trigger_img = np.array([0, 0, 0, 0])
        img = augment_image(trigger_img,train_img,trigger_res,train_res)
    else:
        raise Exception
    return [img, label]


"""
real trigger
    + random -> triggers =  [real], random = True
    + no  -> triggers = [real,[0,0,0,0]], random = False
    + random + no -> triggers = [real,[0,0,0,0]], random = True
    + triggers


"""
"""def generate_train_sample(triggers, train_img, trigger_res: tuple = None, train_res: tuple = None,
                          random_trigger=False) -> list:
    if random_trigger:
        pass
    number_of_triggers = len(triggers)# +1 is no trigger

    if number_of_triggers == 1:
        prob = [1]
    else:
        prob = (number_of_triggers - 1) * [0.5 / (number_of_triggers - 1)]
        prob.insert(0, 0.5)
    # print("prob",prob)
    # print(number_of_triggers)
    choice = choices(range(len(prob)), prob)[0]
    # print("choice: ",choice)

    # correct false is 1, correct is 0
    label = 0 if choice >= 1 else 1
    # print("label: ",label)
    # print(label == 1 & choice != 0)
    assert ((label == 1 and choice == 0) or (label == 0 and choice != 0))
    if random_trigger:
        trigger_img = generate_random_trigger(trigger_res)
    else:
        trigger_img = triggers[choice]
    img = augment_image(trigger_img, train_img, trigger_res, train_res)
    return [img, label]"""