#import neccessary packages
from imutils import paths
import numpy as np
import imutils
import argparse
import pickle
import os
import cv2

#Argument Parsers

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
    help = "Path to input directory of faces + Images")
ap.add_argument("-e", "--embeddings", required=True,
    help = "path to reduces serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
    help = "path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
    help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#load our serializing face dataset from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt.txt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#Load serializer face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

#Get image path to the dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

#initialize our list of extracted facial embeddings and thier names
knownEmbeddings = []
knownNames = []

#initialize the total number of faces processed
total = 0

#Looping over image path
for (i, imagePath) in enumerate(imagePaths):
    #extract the persons name from the image path
    print(f"[INFO] processing image {i+1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]


    # load the image, resize it to have a width of 600 pixel (while maintaining the aspect ratio), and then grab the image dimension
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h,w) = image.shape[:2]

    #construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300,300)), 1.0, (300,300),
        (104.0, 177.0, 123.0), swapRB = False, crop=False)

    #apply OpenCV's deep learning-based face detector to localize faces in the imput image
    detector.setInput(imageBlob)
    detections = detector.forward()

    #ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0,0,:,2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob=cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB = True, crop = False)
            embedder.setInput(faceBlob)
            vec=embedder.forward()
            # add the name of the person + corresponding face
            # embedding to their respective lists
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data={"embeddings": knownEmbeddings, "names": knownNames}
f=open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()





