"""
 Module to detect and crop faces from photo:
    1. Detect faces in photo
    2. Align face to horizontal
    3. Crop face region
    4. Resize cropped face to 48x48
"""


import cv2
import copy
import numpy as np
from PIL import Image


def detect_face(name, mode):
    # print instructions
    if mode == 'auto':
        print('Auto detect mode selected.\nPlease be advised: If the result is satisfiable, '
              'try manual mode and type in parameters to get better result.')
    elif mode == 'manual':
        print('Manual detect mode selected\nType s to set scaleFactor, type m to set minNeighbour, type c to confirm\n'
              'Parameter usages:\nscaleFactor: smaller value increase both the chance of detection and the time cost.\n'
              'minNeighbors: higher value results in less detections but with higher quality')
    else:
        print('ERROR: detect mode is not chosen')
        return None

    # load image
    original = read_image(name)
    # make deep copy of the image
    img = copy.deepcopy(original)
    # convert the image into grayscale
    gray = convert_to_grayscale(img)

    # detect and show faces in the original image
    faces = find_face_regions(img, gray, mode)
    if faces is None:
        return None

    # rotate and crop each detected face
    count = 1
    for (x, y, w, h) in faces:
        # create regions of interest on both original image and grayscale image
        roi_color = copy.deepcopy(original[y:y+h, x:x+w])
        roi_color_copy = copy.deepcopy(roi_color)
        roi_gray = copy.deepcopy(gray[y:y+h, x:x+w])

        # detect eyes in face region
        left_eye, right_eye = find_eyes(roi_color, roi_gray, mode)
        if left_eye is None:
            continue

        # calculate angle and direction for rotation
        angle, direction = calculate_angle(left_eye, right_eye, roi_color)
        if angle is None:
            continue

        # rotate face region
        rotated = rotate(roi_color_copy, angle, direction)

        # crop face region
        cropped = crop_face(rotated, mode)
        if cropped is None:
            continue

        # resize image to 48 x 48
        resized = resize_image(cropped, 48, 48)

        # save result to file
        cv2.imwrite(name + '.face' + str(count) + '.jpg', resized)
        count += 1


def read_image(name):
    # load and show image
    img = cv2.imread(name)
    show_image('original image', img)

    return img


def show_image(window_name, img):
    cv2.imshow(window_name, img)
    # wait for user to press any key, this is necessary to avoid python kernel form crashing
    cv2.waitKey(0)
    # close all open windows
    cv2.destroyAllWindows()


def convert_to_grayscale(img):
    # convert the image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray


def find_face_regions(img, gray, mode):
    # create face cascade
    face_cascade = cv2.CascadeClassifier("detect_face/haarcascade_frontalface_default.xml")

    # detect faces in image
    # parameters for detectMultiScale
    scaleFactor = 1.1
    minNeighbors = 5
    # timeout count in case of infinite while loop
    timeout = 0
    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)

    # no faces detected in image
    if mode == 'auto':
        while len(faces) == 0:
            # time out, failed to detect faces
            if timeout > 10:
                print("ERROR: timeout, failed to detect correct faces(s) in given image!")
                return None

            # adjust parameters and detect again
            else:
                timeout += 1
                minNeighbors -= 1
                faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)

    else:
        while True:
            # show current result
            img_copy = copy.deepcopy(img)
            for (x, y, w, h) in faces:
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
            show_image('current result', img_copy)

            # let user chose what to do
            print('Please choose an option:')
            ipt = input()

            # adjust parameters and detect again
            if ipt == 's':
                print('Please type the value you want to set for scaleFactor (bigger than 1):\nCurrent value:' + str(scaleFactor))
                ipt = input()

                if float(ipt) > 1:
                    scaleFactor = float(ipt)
                    print('New value: ' + str(ipt) + '\n')
                    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
                else:
                    print('Useless input, scaleFactor must bigger than 1!\n')

            # adjust parameters and detect again
            elif ipt == 'm':
                print('Please type the value you want to set for minNeighbors (must be integer and bigger than or '
                      'equals to 1):\nCurrent value:' + str(minNeighbors))
                ipt = input()

                if float(ipt) >= 1 and float(ipt) % 1 == 0:
                    minNeighbors = int(ipt)
                    print('New value: ' + str(ipt) + '\n')
                    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
                else:
                    print('Useless input, minNeighbors must be integer and bigger than or equals to 1!\n')

            # finish detection
            elif ipt == 'c':
                print('Face detection finished\n')
                break

            # useless input
            else:
                print('Unknown option, please choose from \'s\', \'m\', \'c\'\n')

    # draw rectangles around faces and show the result
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    show_image('faces detected', img)

    return faces


def find_eyes(roi_color, roi_gray, mode):
    # create eye cascade
    eye_cascade = cv2.CascadeClassifier("detect_face/haarcascade_eye.xml")

    # detect eyes in image
    # parameters for detectMultiScale
    scaleFactor = 1.1
    minNeighbors = 5
    # timeout count in case of infinite while loop
    timeout = 0
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor, minNeighbors)

    if mode == 'auto':
        # no eyes detected in given face region
        while len(eyes) == 0:
            # time out, failed to detect eyes
            if timeout > 10:
                print("ERROR: timeout, failed to detect correct eye(s) in given face region!")
                return None, None

            # adjust parameters and detect again
            else:
                timeout += 1
                minNeighbors -= 1
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor, minNeighbors)

        # more than 2 eyes detected in given face region
        while eyes.shape[0] > 2:
            # time out, failed to detect eyes
            if timeout > 10:
                print("ERROR: timeout, failed to detect correct eye(s) in given face region!")
                return None, None

            # adjust parameters and detect again
            else:
                timeout += 1
                minNeighbors += 1
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor, minNeighbors)

        # less than 2 eyes detected in given face region
        while eyes.shape[0] < 2:
            # time out, failed to detect eyes
            if timeout > 10:
                print("ERROR: timeout, failed to detect correct eye(s) in given face region!")
                return None, None

            # adjust parameters and detect again
            else:
                timeout += 1
                minNeighbors -= 1
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor, minNeighbors)

    else:
        while True:
            # show current result
            img_copy = copy.deepcopy(roi_color)
            for (x, y, w, h) in eyes:
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
            show_image('current result', img_copy)

            # let user chose what to do
            print('Please choose an option:')
            ipt = input()

            # adjust parameters and detect again
            if ipt == 's':
                print('Please type the value you want to set for scaleFactor (bigger than 1):\nCurrent value:' + str(
                    scaleFactor))
                ipt = input()

                if float(ipt) > 1:
                    scaleFactor = float(ipt)
                    print('New value: ' + str(ipt) + '\n')
                    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor, minNeighbors)
                else:
                    print('Useless input, scaleFactor must bigger than 1!\n')

            # adjust parameters and detect again
            elif ipt == 'm':
                print('Please type the value you want to set for minNeighbors (must be integer and bigger than or '
                      'equals to 1):\nCurrent value:' + str(minNeighbors))
                ipt = input()

                if float(ipt) >= 1 and float(ipt) % 1 == 0:
                    minNeighbors = int(ipt)
                    print('New value: ' + str(ipt) + '\n')
                    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor, minNeighbors)
                else:
                    print('Useless input, minNeighbors must be integer and bigger than or equals to 1!\n')

            # finish detection
            elif ipt == 'c':
                print('Eye detection finished\n')
                break

            # useless input
            else:
                print('Unknown option, please choose from \'s\', \'m\', \'c\'\n')

    # get region of detected eyes
    index = 0
    for (ex, ey, ew, eh) in eyes:
        if index == 0:
            eye1 = (ex, ey, ew, eh)
        else:
            eye2 = (ex, ey, ew, eh)

        # draw rectangles around detected eyes
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
        index = index + 1
        show_image('eyes detected', roi_color)

    # differentiate between left eye and right eye
    if eye1[0] < eye2[0]:
        left_eye = eye1
        right_eye = eye2
    else:
        left_eye = eye2
        right_eye = eye1

    return left_eye, right_eye


def calculate_angle(left_eye, right_eye, roi_color):
    # calculate coordinates of central points of rectangles around eyes
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

    # draw circles on central points
    cv2.circle(roi_color, left_eye_center, 5, (255, 0, 0), -1)
    cv2.circle(roi_color, right_eye_center, 5, (255, 0, 0), -1)
    cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)

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
    cv2.circle(roi_color, A, 5, (255, 0, 0), -1)

    # draw lines between circles
    cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)
    cv2.line(roi_color, left_eye_center, A, (0, 200, 200), 3)
    cv2.line(roi_color, right_eye_center, A, (0, 200, 200), 3)
    show_image('triangle system between eyes', roi_color)

    # calculate rotation angle
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    if delta_x == 0:
        print("ERROR: division by zero!")
        return None, None
    angle = np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    return angle, direction


def rotate(img, angle, direction):
    # rotate image using warpAffine (scale is wrong)
    # get height and width of the image
    # h, w = img.shape[:2]
    # calculate center point of the image
    # integer division "//"" ensures that we receive whole numbers
    # center = (w // 2, h // 2)
    #
    # # get rotation matrix M using cv2.getRotationMatrix2D
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # apply the rotation to our image using cv2.warpAffine
    # rotated = cv2.warpAffine(img, M, (w, h))

    # rotate image using Pillow
    img1 = Image.fromarray(img)
    rotated = np.array(img1.rotate(angle))

    return rotated


def crop_face(rotated, mode):
    # deep copy of rotated image (to show detected face)
    rotated_copy = copy.deepcopy(rotated)
    # convert the image into grayscale
    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    # detect face in the image (should be 1)
    face = find_face_regions(rotated_copy, rotated_gray, mode)
    # error checking
    if face is None:
        return None
    if face.shape[0] > 1:
        return None

    # crop image according to rectangle around the detected face
    for (x, y, w, h) in face:
        cropped = rotated[x:x+w, y:y+h]
        show_image('cropped face', cropped)

    return cropped


def resize_image(img, w, h):
    resized = cv2.resize(img, (w, h))
    return resized


def test_detect_face():
    # # images with no face
    # detect_face('detect_face/no1.jpg', 'auto')
    # detect_face('detect_face/no2.jpg', 'auto')
    # detect_face('detect_face/no3.jpg', 'auto')
    #
    # # images with incomplete face
    # detect_face('detect_face/incomplete1.jpg', 'auto')
    # detect_face('detect_face/incomplete2.jpg', 'auto')
    # detect_face('detect_face/incomplete3.jpg', 'auto')
    #
    # # images with single face
    detect_face('detect_face/single1.jpg', 'auto')
    detect_face('detect_face/single2.jpg', 'auto')
    detect_face('detect_face/single3.jpg', 'auto')

    # images with multi faces
    # detect_face('detect_face/multi1.jpg', 'manual')
    # detect_face('detect_face/multi2.jpg', 'auto')
    # detect_face('detect_face/multi3.jpg', 'auto')


# main program
if __name__ == '__main__':
    test_detect_face()
