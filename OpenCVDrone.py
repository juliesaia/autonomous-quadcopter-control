# organize imports
import cv2
import imutils
import numpy as np
import speech_recognition as sr
from vjoy import vj, setJoy


# OpenCV2 code from https://github.com/Gogul09/gesture-recognition (segment.py)

##############
###### params
#############
steps = 1 #average over n frames, from 1 to 30
scale = 12000 #sensitivity, from 0 to 16000
##############
##############
##############

# global variables
dh = 1 #keep at 1 for joystick -1 to 1
desiredheight = 0 #keep at 0 for joystick default pos 0
global thresholded
global hand
bg = None
elsecount = 0
calibcount = 0
startcount = 0
heightcalib = 5
bgcalibstart = 0
bgcalibend = 0
averageheight = []
calibrateflag = False
startflag = False

vj.open()

xPos = 0
yPos = desiredheight



#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 300, 590

    # initialize num of frames
    num_frames = 0
    print("Press 's' to start")
    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        cv2.imshow("Video Feed", clone)
        
        

        if not startflag:
            if startcount < steps:
                startcount += 1
            else:
                yPos = desiredheight
                startcount = 0
        if (keypress == ord('s')):
            bgcalibstart = num_frames
            bgcalibend = num_frames + 30
            startflag = True
        
        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if (bgcalibstart <= num_frames and num_frames <= bgcalibend and num_frames != 0):
            if (num_frames == bgcalibstart):
                print("Running average... Wait 1 second")
            if (num_frames == bgcalibend):
                print('Done!')
                print("Press 'c' to calibrate")
            run_avg(gray, aWeight)
        
        else:
            
            if (bgcalibstart != 0 or bgcalibend != 0):
                
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

                    # draw the segmented region and display the frame
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    
                    # dimensions = thresholded.shape
            
                    # height, width, number of channels in image
                    height = thresholded.shape[0]
                    width = thresholded.shape[1]
                    rows = []
                    
                    rowtotal = 0
                    highestrow = 0
                    for i in range(height):
                        for j in range(width):
                            # add 255 if white, 0 if black
                            rowtotal += thresholded[i][j]
                        #append average value of row
                        rows.append(rowtotal/width)
                        rowtotal = 0
                    
                    middlecol = False
                    for i in range(height):
                        if thresholded[i][width//2] == 255:
                            middlecol = True
                            break
                    
                    ########################################
                    #### WIDEST ROW
                    #######################################
                    
                    if calibrateflag and middlecol: #only do things if calibrated and hand in the middle
                        #if hand below calib
                        if ((heightcalib - rows.index(max(rows))) < 0):
                            #dh * the diff in the most white row (middle of hand) and heightcalib normalized to the diff from heightcalib to bottom (height) + desiredheight
                            if (len(averageheight) < steps):
                                averageheight.append(dh * (heightcalib - rows.index(max(rows)))/abs(heightcalib-height) + desiredheight)
                            else:
                                yPos = sum(averageheight)/len(averageheight)
                                averageheight = []
                        else:
                            #dh * the diff in the most white row (middle of hand) and heightcalib normalized to the diff from heightcalib to top (0) + desiredheight
                            if (len(averageheight) < steps):
                                averageheight.append(dh * (heightcalib - rows.index(max(rows)))/abs(heightcalib) + desiredheight)
                            else:
                                yPos = sum(averageheight)/len(averageheight)
                                averageheight = []
                    else:
                        if (calibcount < steps):
                            calibcount += 1
                        else:
                            yPos = 0
                            calibcount = 0
                    
                    if calibrateflag:
                        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
                        if not (calibrateflag and middlecol):
                            cv2.line(thresholded, (0,heightcalib), (width-1, heightcalib), (0,255,0), 4)
                        else:   
                            cv2.line(thresholded, (0,heightcalib), (width-1, heightcalib), (0,255,0), 4)
                            cv2.line(thresholded, (0,rows.index(max(rows))), (width-1, rows.index(max(rows))), (0,0,255), 4)
                    cv2.imshow("Thresholded", thresholded)
                    if calibrateflag:
                        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)
        

                else:
                    if (elsecount < steps):
                        elsecount += 1
                    else:
                        yPos = desiredheight
                        elsecount = 0
                    



                
        
        num_frames += 1


        # if the user pressed "q", then stop looping
        
        if keypress == ord("q"):
            break



        if keypress == ord("c"):
            (thresholded, segmented) = hand

            # draw the segmented region and display the frame
            
            # dimensions = thresholded.shape
    
            # height, width, number of channels in image
            height = thresholded.shape[0]
            width = thresholded.shape[1]
            rows = []
            rowtotal = 0
            for i in range(height):
                for j in range(width):
                    rowtotal += thresholded[i][j]
                rows.append(rowtotal/width)
                rowtotal = 0
            #most white row
            middlecol = False
            for i in range(height):
                if thresholded[i][width//2] == 255:
                    middlecol = True
                    break
            if middlecol:
                heightcalib = rows.index(max(rows))
                calibrateflag = True
                
                print("Calibrated!")
                print("Press 'q' to quit.") 
            else:
                print("Error: make sure you hand is centered in frame.")


            
        setJoy(xPos, yPos, scale)
# free up memory
camera.release()
cv2.destroyAllWindows()
# calling this function requests that the background listener stop listening
#stop_listening(wait_for_stop=False)
setJoy(0, 0, scale)
vj.close()