import cv2
import dlib
from math import hypot


def get_points(prediction,image,object, number):
    points = prediction(image, object)
    point = (points.part(number).x, points.part(number).y)
    return point


def replaced_obj(img,frame,width,height,point_a):
    resized_image = cv2.resize(img,(width,height))
    resized_image_gray = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(resized_image_gray, 25, 255, cv2.THRESH_BINARY_INV)
    object_area = frame[point_a[1]:point_a[1]+height,
                  point_a[0]:point_a[0]+width]
    cutted_out_object = cv2.bitwise_and(object_area, object_area, mask=mask)
    merged = cv2.add(cutted_out_object, resized_image)
    frame[point_a[1]:point_a[1] + height,
    point_a[0]:point_a[0] + width] = merged
    return  merged

def height_eye(bottom,top,multiplier):
    height = int(hypot(bottom[0] - top[0],
              bottom[1] - top[1]) * multiplier)
    return height

def width_eye(left_point,right_point,multiplier):
    width = int(hypot(left_point[0] - right_point[0],
            left_point[1] - right_point[1]) * multiplier)
    return width

def height_brow(left_point,top):
    height = int(hypot(left_point[0]-top[0],
                        left_point[1] - top[1]))
    return height

def width_brow(left_point, right_point):
    width = int(hypot(right_point[0]-left_point[0],
                         right_point[1]-left_point[1]))
    return width