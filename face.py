import cv2
import numpy as np
import dlib
from math import hypot
import detect

camera = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
face_pose_prediction = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

nose_img = cv2.imread("clownnose.png")
left_eye_img = cv2.imread("e11.png")
right_eye_img = cv2.imread("eye2.png")
left_brow_img = cv2.imread("neon.png")
right_brow_img = cv2.imread("right_brow (2).png")

while True:
    #frame by frame
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blank = np.zeros_like(gray)
    faces = face_detector(gray)
    a = cv2.merge([blank, blank, gray])
    #find points in face
    for face in faces:
        #contrast = cv2.addWeighted(gray, 2.5, np.zeros(gray), 0, 0)
        #a = cv2.merge([blank,blank,gray])
        #nose
        tip_of_nose = detect.get_points(face_pose_prediction,gray,face,29)
        left_side_nose = detect.get_points(face_pose_prediction,gray,face,31)
        right_side_nose = detect.get_points(face_pose_prediction,gray,face,35)
        center_nose = detect.get_points(face_pose_prediction,gray,face,30)

        width_nose = int(hypot(left_side_nose[0] - right_side_nose[0],
                               left_side_nose[1] - right_side_nose[1])*1.6)
        height_nose = int(0.991 * width_nose)

        left_up_corner = (int(center_nose[0] - width_nose/2),
                             int(center_nose[1] - height_nose/2))
        bottom_right_corner = (int(center_nose[0]+width_nose/2),
                       int(center_nose[1]+height_nose/2))

        detect.replaced_obj(nose_img,a,width_nose,height_nose,left_up_corner)
        #left eye
        left_eye_left_side = detect.get_points(face_pose_prediction,gray,face,36)
        left_eye_right_side = detect.get_points(face_pose_prediction, gray, face, 39)
        left_eye_top = detect.get_points(face_pose_prediction,gray,face,37)
        left_eye_bottom = detect.get_points(face_pose_prediction, gray, face, 41)

        width_left_eye = detect.width_eye(left_eye_left_side, left_eye_right_side,1.6)
        height_left_eye = detect.height_eye(left_eye_bottom,left_eye_top,7)
        left_up_left_eye_corner = (int(left_eye_left_side[0]-7),
                                   int(left_eye_left_side[1]-height_left_eye/2))
        #bottom_right_left_eye_corner = (int(left_eye_right_side[0]),
                       #int(left_eye_right_side[1]+height_nose/2))
        detect.replaced_obj(left_eye_img,a,width_left_eye,height_left_eye,left_up_left_eye_corner)

        #right eye
        right_eye_left_side = detect.get_points(face_pose_prediction, gray, face, 42)
        right_eye_right_side = detect.get_points(face_pose_prediction, gray, face, 45)
        right_eye_top = detect.get_points(face_pose_prediction, gray, face, 44)
        right_eye_bottom = detect.get_points(face_pose_prediction, gray, face, 46)

        width_right_eye = detect.width_eye(right_eye_left_side,right_eye_right_side,1.6)
        height_right_eye = detect.height_eye(right_eye_bottom,right_eye_top,7)
        left_up_right_eye_corner = (int(right_eye_left_side[0]-7),
                                   int(right_eye_left_side[1] - height_right_eye / 2))
        detect.replaced_obj(right_eye_img, a, width_right_eye, height_right_eye, left_up_right_eye_corner)

        #left brow
        left_brow_left_side = detect.get_points(face_pose_prediction, gray, face, 17)
        left_brow_right_side = detect.get_points(face_pose_prediction, gray, face, 21)
        left_brow_top = detect.get_points(face_pose_prediction, gray, face, 19)

        width_left_brow = detect.width_brow(left_brow_left_side,left_brow_right_side)
        height_left_brow = detect.height_brow(left_brow_left_side,left_brow_top)*2
        left_up_left_brow_corner = (int(left_brow_left_side[0]+15),int(left_brow_left_side[1]-(height_left_brow*1.3)))
        detect.replaced_obj(left_brow_img, a, width_left_brow*2, height_left_brow, left_up_left_brow_corner)
        #right brow
        right_brow_left_side = detect.get_points(face_pose_prediction, gray, face, 22)
        right_brow_right_side = detect.get_points(face_pose_prediction, gray, face, 26)
        right_brow_top = detect.get_points(face_pose_prediction, gray, face, 24)

        width_right_brow = detect.width_brow(right_brow_left_side, right_brow_right_side)
        height_right_brow = detect.height_brow(right_brow_left_side, right_brow_top)
        left_up_right_brow_corner = (int(right_brow_left_side[0]), int(right_brow_left_side[1] - (height_right_brow * 0.5)))
        detect.replaced_obj(right_brow_img, a, width_right_brow, height_right_brow, left_up_right_brow_corner)

        #cv2.imshow('a',a)
    cv2.imshow('frame', a)
    #cv2.imshow('con',contrast)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()