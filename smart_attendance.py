
# import the necessary packages
from imutils.video import VideoStream
from pyzbar import pyzbar
import datetime
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import dlib
import cv2
import h5py
#import _pickle as cPickle
import pickle
import face_recognition

from firebase import firebase

import os
from PIL import Image, ImageDraw

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-o", "--output", type=str, default="barcodes.csv",
	help="path to output CSV file containing barcodes")
args = vars(ap.parse_args())

folder = "output"
if os.path.exists(folder):
	print ("Folder exist")
else:
	os.mkdir(folder)

with open("trained_knn_model.clf", 'rb') as f:
 	knn_clf = pickle.load(f)

firebase = firebase.FirebaseApplication("https://smart-attendance-1600957380714.firebaseio.com/",None)


# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	
	camera = cv2.VideoCapture(0)
 
# otherwise, grab a reference to the video file
else:

	camera = cv2.VideoCapture(args["video"])

csv = open(args["output"], "w")
found = set()
i=0

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:	

	# grab the current frame
	(grabbed, image1) = camera.read()
	image1 = imutils.resize(image1, width=1000)
	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break
	
	image = image1[:, :, ::-1]

	X_face_locations = face_recognition.face_locations(image)

	barcodes = pyzbar.decode(image1)

	gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(image1, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(image1, [rightEyeHull], -1, (0, 255, 0), 1)	
		if ear < EYE_AR_THRESH:
			COUNTER += 1
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			# reset the eye frame counter
			COUNTER = 0


	# If no faces are found in the image, return an empty result.
	if COUNTER != 0:

		if len(X_face_locations) != 0:
			
			# Find encodings for faces in the test iamge
			faces_encodings = face_recognition.face_encodings(image, known_face_locations=X_face_locations)

			# Use the KNN model to find the best matches for the test face
			print(np.array(faces_encodings).shape)
			closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
			are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(X_face_locations))]

			#draw = ImageDraw.Draw(image)

			# Predict classes and remove classifications that aren't within the threshold
			predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
			for name, (top, right, bottom, left) in predictions:
				
				cv2.rectangle(image1, (left,bottom),(right,top), (0, 255, 0), 2)

				# show the face number
				cv2.putText(image1, "{}".format(name), (left-10, top-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				for barcode in barcodes:

					(x, y, w, h) = barcode.rect
					cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
					barcodeData = barcode.data.decode("utf-8")
					barcodeType = barcode.type
		
					text = "{} ({})".format(barcodeData, barcodeType)
					cv2.putText(image1, text, (x, y - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
					now = datetime.datetime.now()
					data = {
							"Date":now.strftime("%m/%d/%Y"),
							"Time":now.strftime("%H:%M:%S"),
							"ID":barcodeData
						}
					result = firebase.post('/smart-attendance-1600957380714/Data',data)
			
					csv.write("{},{},{}\n".format(now.strftime("%m/%d/%Y"),now.strftime("%H:%M:%S"),
						barcodeData))
					csv.flush()

					found.add(barcodeData)
					frame= os.path.join(folder+"/","Output"+str(i)+ ".jpg")
					cv2.imwrite(frame,image1)
					i+=1

	cv2.imshow("output image",image1)	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# clean up the camera and close any open windows
camera.release()
cv2.destroyAllWindows()