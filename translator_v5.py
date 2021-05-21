import cv2, pickle
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('model_cnn_v5_seg.h5')

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350



def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)



def segment_hand(frame, threshold=10):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    #Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)


def get_pred_text_from_db(pred_class):
	if pred_class <= 25 :
		return chr(pred_class + 65)
	elif pred_class == 26 :
		return "space"
	else:
		return "del"


def text_mode(cam):
	text = ""
	word = ""
	count_same_frame = 0
	num_frames = 0
	percentage = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		old_text = text

		img = cv2.flip(img, 1)
		roi=img[ROI_top:ROI_bottom, ROI_right:ROI_left]
		gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

		if num_frames < 100:
			cal_accum_avg(gray_frame, accumulated_weight)
			cv2.putText(img, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
		else:
			hand = segment_hand(gray_frame)

			if hand is not None:
				thresh, hand_segment = hand
				cv2.drawContours(img, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
				cv2.imshow("Thesholded Hand Image", thresh)

				thresh = cv2.resize(thresh, (40, 40))
				thresh = np.array(thresh, dtype=np.float32)
				thresh = np.reshape(thresh, (1, 40, 40, 1))

				pred_probab = model.predict(thresh)[0]
				pred_class = list(pred_probab).index(max(pred_probab))
				pred_probab = max(pred_probab)
				percentage = round(pred_probab*100, 1)
				
				if pred_probab*100 > 70:
					text = get_pred_text_from_db(pred_class)
                
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0
                
				if count_same_frame > 30:
					if text == "space":
						word = word + " "
						count_same_frame = 0
					elif text == "del":
						n = len(word) - 1
						word = word[:n]
						count_same_frame = 0
					else:
						word = word + text
						count_same_frame = 0
			
			else:
				text = ""
				word = ""
				percentage = 0

		num_frames += 1
        
		blackboard = np.full((480, 1080, 3), 255, dtype=np.uint8)
		cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0,0))
		cv2.putText(blackboard, "Predicted text = " + text + " - Percentage = "+str(percentage)+" %", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
		cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
		cv2.rectangle(img, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)

		res = np.hstack((img, blackboard))
		cv2.imshow("Recognizing gesture", res)
		#cv2.imshow("thresh", thresh)
		keypress = cv2.waitKey(1)
		
		if keypress == ord('q') or keypress == ord('c'):
			break

	if keypress == ord('c'):
		return 2
	else:
		return 0

def recognize():
	cam = cv2.VideoCapture(1)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
		else:
			break
	
recognize()
