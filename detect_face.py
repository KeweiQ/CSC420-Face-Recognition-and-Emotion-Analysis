import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate(name):
    # create face_cascade and eye_cascade objects
    face_cascade=cv2.CascadeClassifier("detect_face/haarcascade_frontalface_default.xml")
    eye_cascade=cv2.CascadeClassifier("detect_face/haarcascade_eye.xml")

    # load image
    img = cv2.imread(name)
    cv2.imshow('window_name', img)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()

    # get height and width of the image
    h, w = img.shape[:2]

    # convert the image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in image
    # 1 face per image for now, implement multi face version later
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # draw rectangles around faces
    for (x, y,  w,  h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow('window_name', img)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()

    # Creating two regions of interest
    # roi_gray = gray[y:(y + h), x:(x + w)]
    # roi_color = img[y:(y + h), x:(x + w)]

    # detect eyes in image
    # 1 pair of eyes per image for now, implement multi face version later
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
    index = 0

    # divide one eye from another
    for (ex, ey,  ew,  eh) in eyes:
        if index == 0:
            eye1 = (ex, ey, ew, eh)
        elif index == 1:
            eye2 = (ex, ey, ew, eh)

        if index < 2:
            # draw rectangles around detected eyes
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
            index = index + 1
            cv2.imshow('window_name', img)
            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)
            # closing all open windows
            cv2.destroyAllWindows()

    # differentiate between left eye and right eye
    if eye1[0] < eye2[0]:
        left_eye = eye1
        right_eye = eye2
    else:
        left_eye = eye2
        right_eye = eye1

    # calculate coordinates of a central points of rectangles around eyes
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

    # draw circles on central points
    cv2.circle(img, left_eye_center, 5, (255, 0, 0), -1)
    cv2.circle(img, right_eye_center, 5, (255, 0, 0), -1)
    cv2.line(img, right_eye_center, left_eye_center, (0, 200, 200), 3)

    # detect the third point of the rectangle
    if left_eye_y > right_eye_y:
        A = (right_eye_x, left_eye_y)
        # integer -1 indicates that the image will rotate in the clockwise direction
        direction = -1
    else:
        A = (left_eye_x, right_eye_y)
        # integer 1 indicates that image will rotate in the counter clockwise direction
        direction = 1

    # draw circle on the third point of the rectangle
    cv2.circle(img, A, 5, (255, 0, 0), -1)

    # draw lines between circles
    cv2.line(img, right_eye_center, left_eye_center, (0, 200, 200), 3)
    cv2.line(img, left_eye_center, A, (0, 200, 200), 3)
    cv2.line(img, right_eye_center, A, (0, 200, 200), 3)
    cv2.imshow('window_name', img)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()

    # calculate rotation angle
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    # calculate center point of the image
    # integer division "//"" ensures that we receive whole numbers
    center = (w // 2, h // 2)

    # get roration matrix M using cv2.getRotationMatrix2D
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # apply the rotation to our image using cv2.warpAffine
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow('window_name', rotated)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()

    # # calculate distance between the eyes in the first image
    # dist_1 = np.sqrt((delta_x * delta_x) + (delta_y * delta_y))
    # # calculate distance between the eyes in the second image
    # dist_2 = np.sqrt((delta_x_1 * delta_x_1) + (delta_y_1 * delta_y_1))
    #
    # # calculate the ratio
    # ratio = dist_1 / dist_2
    # # Defining the width and height
    # h=476
    # w=488
    # # Defining aspect ratio of a resized image
    # dim = (int(w * ratio), int(h * ratio))
    # # We have obtained a new image that we call resized3
    # resized = cv2.resize(rotated, dim)
    # cv2_imshow(resized)
    # # Defining the width and height
    # h=740
    # w=723
    # # Defining aspect ratio of a resized image
    # dim = (int(w * ratio), int(h * ratio))
    # # We have obtained a new image that we call resized3
    # resized = cv2.resize(rotated, dim)
    # cv2_imshow(resized)

# main program
if __name__ == '__main__':
    rotate('detect_face/emily.jpg')