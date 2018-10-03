import cv2
import numpy as np

def extract(image, dest):
    #cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("Out", cv2.WINDOW_AUTOSIZE)
    img = cv2.imread(image, 0)
    #cv2.imshow("Img", img)

    ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(img,ret,2*ret)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 10)

    im2, cnts, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    max_area,index,i = -1,-1,0
    for cnt in cnts:
        if (cv2.contourArea(cnt) > max_area):
            max_area = cv2.contourArea(cnt)
            index = i
        i = i+1

    if (index != -1):
        rect = cv2.minAreaRect(cnts[index])
        box = cv2.boxPoints(rect)

        box = np.int0(box)
        angle = rect[-1]

        rows,cols = img.shape

        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))

        M = np.append(M, [[0,0,1]], axis=0)
        box = np.append(box, [[1],[1],[1],[1]], axis=1).T

        newbox = np.matmul(M, box)
        newbox = newbox[0:-1].T
        newbox = np.int0(newbox)
        dst = dst[newbox[2][1]:newbox[0][1], newbox[1][0]:newbox[3][0]]

        final = cv2.threshold(dst,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        fl,ft,fb,fr = 0,0,0,0
        H,W = final.shape

        for y in range(H):
            for x in range(W):
                cl =  final[y][x]
                if cl == 0:
                    fl = x if fl==0 else min(fl,x)
                    fr = x if fr==0 else max(fr,x)
                    ft = y if ft==0 else min(ft,y)
                    fb = y if fb==0 else max(fb,y)

        final = final[ft:fb, fl:fr]
        cv2.imwrite(dest, final)
        #cv2.imshow("Out", final)
        #cv2.waitKey(0)
#extract("./Testing/img8.jpg", "./Testing/ha.jpeg")