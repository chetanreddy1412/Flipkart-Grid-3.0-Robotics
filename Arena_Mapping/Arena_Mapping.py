import cv2
import numpy as np

#HSV FINDER TOOL

def nothing(x):
    pass

def create_Trackbars(trackbarwindow = 'Controls'):
    
    lower = [0,0,0]
    upper = [180,255,255]
    #HUE
    cv2.createTrackbar('lh_mask1',trackbarwindow,lower[0],179,nothing)
    cv2.createTrackbar('uh_mask1',trackbarwindow,upper[0],179,nothing)
    
    #Saturation
    cv2.createTrackbar('ls_mask1',trackbarwindow,lower[1],255,nothing)
    cv2.createTrackbar('us_mask1',trackbarwindow,upper[1],255,nothing)
    
    #Value
    cv2.createTrackbar('lv_mask1',trackbarwindow,lower[2],255,nothing)
    cv2.createTrackbar('uv_mask1',trackbarwindow,upper[2],255,nothing)
    
    #Same for Mask2
    cv2.createTrackbar('lh_mask2',trackbarwindow,lower[0],179,nothing)
    cv2.createTrackbar('uh_mask2',trackbarwindow,lower[0],179,nothing)    
    
    cv2.createTrackbar('ls_mask2',trackbarwindow,lower[1],255,nothing)
    cv2.createTrackbar('us_mask2',trackbarwindow,lower[1],255,nothing)
    
    cv2.createTrackbar('lv_mask2',trackbarwindow,lower[2],255,nothing)
    cv2.createTrackbar('uv_mask2',trackbarwindow,lower[2],255,nothing)
    
    cv2.createTrackbar('save',trackbarwindow,0,1,nothing)
    #cv2.createTrackbar('mode',trackbarwindow,0,3,nothing)

def get_mask_3d(mask_number,hsv_img,trackbarwindow):#Here, 3d indicates that the shape of the mask is a tuple
    
    lh = cv2.getTrackbarPos('lh_mask{}'.format(mask_number),trackbarwindow)
    uh = cv2.getTrackbarPos('uh_mask{}'.format(mask_number),trackbarwindow)
    ls = cv2.getTrackbarPos('ls_mask{}'.format(mask_number),trackbarwindow)
    us = cv2.getTrackbarPos('us_mask{}'.format(mask_number),trackbarwindow)
    lv = cv2.getTrackbarPos('lv_mask{}'.format(mask_number),trackbarwindow)
    uv = cv2.getTrackbarPos('uv_mask{}'.format(mask_number),trackbarwindow)
    
    lower = np.array([lh,ls,lv])
    upper = np.array([uh,us,uv])
    mask = cv2.inRange(hsv_img,lower,upper)
    mask_3d = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)    
    
    return mask_3d

def visualise_contours(img,contours,thickness=3):
    img_c = img.copy()
    cv2.drawContours(img_c,contours,-1,(255,0,0),thickness)
    plt.imshow(img_c,cmap='gray')
    plt.show()
    return img_c

def get_points_for_warping(qmask):
    
    qmask_gray = cv2.cvtColor(qmask,cv2.COLOR_BGR2GRAY)
    contours,h = cv2.findContours(qmask_gray.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    max_area_cnt = max(contours, key = cv2.contourArea)
    peri = cv2.arcLength(max_area_cnt, True)
    approx = cv2.approxPolyDP(max_area_cnt, 0.02 * peri, True)
    pnts = approx.squeeze()
    
    if pnts.shape !=(4,2):
        visualise_contours(img,[contours])
    return pnts,contours

def order_points(pts):
    # initialise a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
#*************************************************************#


    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
#returns a warped image
#parameters: image and the 4 pts

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def initiate_mapping(cap):
    
    if cap.isOpened()==False:
        print('Could not access feed')
        return
    
    trackbarwindow = 'Controls'
    cv2.namedWindow(trackbarwindow)
    create_Trackbars(trackbarwindow)
    
    pnts = np.array([[0,0],[0,1],[1,0],[1,1]]) 
    while True:

        ret,img = cap.read()
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        save = cv2.getTrackbarPos('save',trackbarwindow)


        mask1 = get_mask_3d(1,hsv_img,trackbarwindow)
        mask2 = get_mask_3d(2,hsv_img,trackbarwindow)
        mask = cv2.bitwise_or(mask1,mask2)

        mask1_and_mask2 = np.hstack((mask1,mask2))
        img_and_mask = np.hstack((img,mask))
        
        stacked = np.vstack((img_and_mask,mask1_and_mask2))
        stacked = cv2.resize(stacked,None,fx=0.25,fy=0.25)
        cv2.imshow("Mask",stacked)
        
        mask_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        contours,h = cv2.findContours(mask_gray.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_area_cnt = max(contours, key = cv2.contourArea)
            peri = cv2.arcLength(max_area_cnt, True)
            approx = cv2.approxPolyDP(max_area_cnt, 0.02 * peri, True)
            pnts = approx.squeeze()

            img_contours = img.copy()
            cv2.drawContours(img_contours,[pnts],-1,(0,0,255),15)
            img_contours = cv2.resize(img_contours,None,fx = 0.7,fy=0.7)
            cv2.imshow('Grid Recognition',img_contours)
        
        else:
            cv2.imshow('Grid Recognition',img.copy())


        warped = four_point_transform(img,pnts)
        warped = cv2.resize(warped,None,fx = 0.7,fy=0.7)
        cv2.imshow('Warped',warped)
        key = cv2.waitKey(10)
        
        if save ==1:
            mask_no = np.random.randint(1,10000)
            cv2.imwrite('mask_{}.jpg'.format(mask_no),mask)
            break
        if key == ord('s'):
            break
            
    cv2.destroyAllWindows()
    while cap.isOpened():
        ret,img = cap.read()
        
        warped = four_point_transform(img,pnts)
        cv2.imshow('Grid',warped)
        
        if cv2.waitKey(10)==ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.release()

cap = cv2.VideoCapture(0)
initiate_mapping(cap)

