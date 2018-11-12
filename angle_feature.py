import cv2
import numpy as np
import math

omega = 1

def angle(image):
    img = cv2.imread(image, 0)
    H,W = img.shape
    #print(H,W)
    #deviding the image into 7*5 grids
    y,x=7,5

    step_y, step_x = math.ceil(H/y),math.ceil(W/x)
    angle_mat = np.zeros((y,x))

    for Y in range(0,y):
        for X in range(0,x):
            col = Y*step_y
            row = X*step_x
            ref_x,ref_y = row,col+step_y
            angle = 0
            no_points = 0
            for c in range(col,col+step_y):
                for r in range(row,row+step_x):
                    if c<H and r<W:
                        if img[c][r] == 0:
                            no_points = no_points+1
                            act_x,act_y = r,c
                            angle = (angle+math.atan((ref_y-act_y)/(act_x-ref_x))) if (act_x-ref_x) != 0 else math.radians(angle+90)
            angle_mat[Y][X] = omega*(math.sin(angle/no_points)) if no_points != 0 else 0
    '''
    for el in angle_mat:
        print(el)
    #print(len(angle_mat), len(angle_mat[0]))

    cv2.imshow("Angle", angle_mat)
    cv2.waitKey(0)
    '''
    angle_mat = angle_mat.flatten()
    return angle_mat
