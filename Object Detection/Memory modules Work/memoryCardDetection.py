# import the necessary packages
from contextlib import contextmanager
from difflib import context_diff
from operator import contains
import os, random, cv2, yaml, re
from platform import architecture
from pickle import FALSE
from tkinter import image_names
from sys import flags
from PIL import Image
import numpy as np
from glob import glob
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split

#use the image in the folder img to obtain dataset of cropped image, each image give 25 cropped images
def extractDatas(path):
    if (os.path.exists('calibration.yaml') == False):
        print("Calibration.yaml not found... Starting calibrating camera")
        #load images
        imgs = load_images_from_folder("boardImg")

        #calculate camera data
        cameraCalibration(imgs)
    
    folders = os.listdir(path)
    tot = 0
    for folder in folders:
        if os.path.isdir(path + '/' + folder):
            print("Starting treating " + path + '/' + folder)
            for filename in os.listdir(path +'/' + folder):
                tot +=1
                img = cv2.imread(os.path.join(path + '/' + folder, filename))
                if img is not None:
                    #get image number
                    i = [float(s) for s in re.findall(r'-?\d+\.?\d*', filename)]
                    i = int(i[0])
                    print("Starting extracting data from img " + str(i))
                    #load dictionnary and params
                    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
                    arucoParam = cv2.aruco.DetectorParameters_create()

                    #load matrix and distortion coefficient from yaml file
                    with open('calibration.yaml', 'r') as f:
                        loadeddict = yaml.safe_load(f)
                    mtx = np.array(loadeddict.get('camera_matrix'))
                    dist_coeff = np.array(loadeddict.get('dist_coeff'))

                    # get the undistorted image
                    h,w = img.shape[:2]
                    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coeff, (w,h), 1, (w,h))
                    undistorted_img = cv2.undistort(img, mtx, dist_coeff, None, newcameramtx)

                   # crop the image
                    x, y, w, h = roi
                    undistorted_img = undistorted_img[y:y+h, x:x+w]
                    
                    #find 2D and 2D of corners
                    aruCo2D, aruCo3D= findCoordCorners(tot, filename, undistorted_img, arucoDict, arucoParam)

                    if aruCo2D is not None:
                        try:
                            if len(aruCo2D) >= 4:
                                #initialise 3D position of blocs that may contains Memory Cars
                                blocs3D = get_blocs3D()

                                #initialise 2D position of blocs that may contains Memory Cars
                                blocs2D = get_blocs2D(tot, blocs3D, aruCo2D, aruCo3D, mtx, dist_coeff, undistorted_img)
                            else:
                                print("Unable to found more than 3 arUco, file deleted : ", path + '/' + folder + "/" + filename)
                                os.remove(path + '/' + folder + "/" + filename)
                        except:
                            print("Failed to print points on image, extracting on image " + filename + " canceled")
                            continue
                        #calculation of each spaces between cards
                        #extractDatasImage(blocs2D, undistorted_img, folder, filename)
                    else:
                        print("Unable to find more than 2 arUCo, image deleted")
                        os.remove(path + '/' + folder + "/" + filename)
                else:
                    print("File ignored and deleted : " + path + '/' + folder + "/" + filename)
                    os.remove(path + '/' + folder + "/" + filename)
                

#load all images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#get Camera datas
def cameraCalibration(imgs):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (7,7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 

    # Defining the world coordinates for 3D points
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
    
    i = 0
    for img in imgs:
        print("Treating image ", i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            imgChessBoard = img.copy()
            imgChessBoard = cv2.drawChessboardCorners(imgChessBoard, CHECKERBOARD, corners2, ret)
            cv2.imwrite("./chessBoardCorners/drawnCorners_" + str(i) + ".jpg",imgChessBoard)
        i += 1

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    i = 0
    for img in imgs:
        h,w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        undistorted_img = undistorted_img[y:y+h, x:x+w]
        cv2.imwrite("./undistortedChess/undistorted_" + str(i) + ".jpg", undistorted_img)
        i +=1

    print("Camera matrix is \n", mtx, "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)

#return 2D and 3D coordinates of arucos
def findCoordCorners(i, filename, img, arucoDict, arucoParam):
    #load 3D coordinates
    with open('dataMemoryTray.yaml', 'r') as f:
        datas = yaml.safe_load(f)
    arUco3D = np.array(datas.get('aruCos'))

    # Detect the markers# Read in the image.
    cv2.imwrite("original_img.jpg", img)
    corners, ids, _ = cv2.aruco.detectMarkers(img, arucoDict, parameters = arucoParam)
    imgArucoD = img.copy()
    cv2.aruco.drawDetectedMarkers(imgArucoD, corners, ids)
    cv2.imwrite("./ARUCOS/aruco_detected" + str(i) + ".jpg", imgArucoD)

    arUco2D = np.zeros((4,2))
    corners = np.squeeze(corners)

    if ids is not None:
        if len(ids) < 3:
            print("Not found more than 2 arucos, file ignored :" + filename)
        else:
            #sort the corners with the help of the ids and delete the third corners if there is more than 3
            print("Found at least 3 corners")
            ids = ids.squeeze()
            for i in range(len(ids)):
                arUco2D[ids[i]-1] = corners[i][0]
            if len(ids) < 4:
                index = np.where(arUco2D == 0)
                arUco2D = np.delete(arUco2D, index[0][0], axis=0)
                arUco3D = np.delete(arUco3D, index[0][0], axis=0)
            return arUco2D, arUco3D
    else:
        print("Not found any arucos, file ignored :" + filename)
    return None, None

#we assume that we have the top left is 0,0 in our world and that we have the necesseray data of our box
def get_blocs3D():
    #a table of 25 blocs with each top left point and right bottom point in 3D
    blocs = np.zeros((100, 3), dtype=np.float32)

    #we load our datas
    with open('dataMemoryTray.yaml', 'r') as f:
        datas = yaml.safe_load(f)
    X = np.array(datas.get('xStart')).squeeze()
    Y = np.array(datas.get('yStart')).squeeze()
    step = 9.9
    step2 = 5.0
    for i in range(25):
        blocs[i] = X - [step2,0,0]
        blocs[i+25] = X + [step2,0,0]
        blocs[i+50] = Y - [step2,0,0]
        blocs[i+75] = Y + [step2,0,0]
        X += [step,0,0]
        Y += [step,0,0]
        
    return blocs

#for solveP3P at least 4 points in 3D and 2D must be given
def get_blocs2D(i, blocs3D, aruCo2D, aruCo3D, mtx, dist_coeff, img):
    blocs3D = np.array(blocs3D, dtype=np.double)

    # Find the rotation and translation vectors.
    ret, rvecs, tvecs = cv2.solvePnP(aruCo3D, aruCo2D, mtx, dist_coeff)

    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(blocs3D, rvecs, tvecs, mtx, dist_coeff)

    #draw the poiints projected on the image and save the image in the AXIS folder
    my_drawpoints(img, imgpts, "./AXIS/drawnPointsimgPts" + str(i) + ".jpg")

    #draw the arucos 3D projected points and save the image in the AXIS folder
    #imgptsC, jac = cv2.projectPoints(aruCo3D, rvecs, tvecs, mtx, dist_coeff)
    #my_drawpoints(img, imgptsC, "./AXIS/drawCorners" + str(i) + ".jpg")

    #draw the axis on the image and save the image in the AXIS folder
    axis = np.float32([[300,0,0], [0,300,0], [0,0,300]]).reshape(-1,3)
    imgptsB, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist_coeff)
    my_drawaxis(0, img, aruCo2D, imgptsB, "./AXIS/drawnAxisCorners" + str(i) + ".jpg")

    return imgpts

#drawn axis on image
def my_drawaxis(i, img, corners, imgpts, name):
    imageDrawn = img.copy()
    corners = np.array(corners, dtype=np.int32)
    imgpts = np.array(imgpts, dtype=np.int32)
    corner = tuple(corners[i].ravel())
    imageDrawn = cv2.line(imageDrawn, corner, tuple(imgpts[0].ravel()), (255,0,0), 8)
    imageDrawn = cv2.line(imageDrawn, corner, tuple(imgpts[1].ravel()), (0,255,0), 8)
    imageDrawn = cv2.line(imageDrawn, corner, tuple(imgpts[2].ravel()), (0,0,255), 8)
    cv2.imwrite(name, imageDrawn)

#draw points on image
def my_drawpoints(img, points, name):
    imageDrawn = img.copy()
    points = np.squeeze(points)
    for p in points:
        center = (int(p[0]), int(p[1]))
        imageDrawn = cv2.circle(imageDrawn, center, radius=12, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(name, imageDrawn)
    
#extract in a csv all the cropped image from the image
def extractDatasImage(blocs2D, img, origin, filename):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    #now we need to fix 2 points to loop on the spaces, a loop of 25

    nbImg = len(os.listdir('./cropped_imgs/' + origin))
    blocs2D = np.array(blocs2D, dtype=np.int32)
    for i in range(0,25):
        #with our 4 points we extract a rctangle 
        x1, y1 = np.array(blocs2D[i]).squeeze()
        x2, y2 = np.array(blocs2D[i+25]).squeeze()
        x3, y3 = np.array(blocs2D[i+50]).squeeze()
        x4, y4 = np.array(blocs2D[i+75]).squeeze()

        cnt = np.array([[[x1, y1]],
                [[x2, y2]],
                [[x3, y3]],
                [[x4, y4]]
                ])

        # find the rotated rectangle enclosing the contour
        rect = cv2.minAreaRect(cnt)

        #cropped image
        cropped_img = crop_rect(img, rect)
        cv2.imwrite("./cropped_imgs/" + origin + "/cropped_" + str(i + nbImg) + ".jpg", cropped_img)


# this function is base on post at https://goo.gl/Q92hdp
def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    rows, cols = img.shape[0], img.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    out = cv2.getRectSubPix(img_rot, size, center)
    return out

def renameImages(path):
    os.getcwd()
    folders = os.listdir(path)
    for folder in folders:
        i = 0
        if os.path.isdir(path + '/' + folder):
            for i, filename in enumerate(os.listdir(path + '/' + folder + "/")):
                os.rename(path + '/' + folder + "/" + filename, path + '/' + folder + "/imgTest" + str(i) + ".jpg")

#found all images to generate the dataset
def loadAndPreprocess(folder, origin):
    print("Starting generating dataset from " + folder)
    memoryCards = []
    negCards = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            i = [float(s) for s in re.findall(r'-?\d+\.?\d*', filename)]
            i = int(i[0])
            img = np.asarray(img)
            img = cv2.cvtColor(cv2.resize(img,(16,245)),cv2.COLOR_RGB2GRAY).ravel()
            if origin == "bottom":
                indexArray = np.array([5,11,17,23,27,30,49,50,52,55,74,75,77,80,99,100,102,105,124,125,130,136,142,148,155,
                161,167,173,180,186,192,198,202,205,224,225,230,236,242,248,252,255,274,275,277,280,299,300,305,311,317,323,
                327,330,349,350,353,366,372,375])
            elif origin == "top":
                indexArray = np.array([0,17,23,24,46,47,48,49,50,52,73,74,75,92,98,99,100,117,123,124,146,147,148,149,
                171,172,173,174,175,183,192,199,221,222,223,224,225,233,242,249,271,272,273,274,296,297,298,299,321,322,
                323,324,346,347,348,349,371,372,373,374])
            elif origin == "left":
                indexArray = np.array([5,11,17,25,29,30,50,55,61,67,75,80,86,91,99,105,111,117,125,129,130,150,154,155,
                175,180,186,191,199,205,211,217,225,229,230,250,258,263,269,275,280,286,49,149,174,249,292,299,305,311,
                317,324,329,330,349,350,354,355,374,375,380,386,392,399,405,411,417,425,430,436,442,450,454,455,474,475,
                480,486,492,500,504,505,524,525])
            elif origin == "right":
                indexArray = np.array([3,4,24,25,28,29,49,50,53,54,74,75,79,85,94,100,103,104,124,125,129,135,144,150,154,
                160,169,175,179,185,194,200,204,210,219,225,229,235,244,250,254,260,269,275,278,282,287,294,304,310,319,325,
                328,329,349,350,354,360,369,375,378,379,399,400,403,404,424,425,429,435,444,450,454,460,469,475])
            elif origin == "bottom right":
                indexArray = np.array([0,3,23,24,25,28,48,49,50,53,73,74,75,78,98,99,100,103,123,124,125,128,148,149,150,
                156,171,174,175,178,198,199,200,203,223,224,225,228,248,249])
            elif origin == "bottom left":
                indexArray = np.array([4,23,29,35,40,48,51,54,73,74,76,79,98,99,101,104,123,124,126,129,148,149,151,154,
                173,174,180,181,182,198,204,210,216,223,225,229,248,249,254,260,266,273,279,285,291,298,301,304,323,324,
                326,348,349,351,354,374,373,379,385,391,398,401,404,423,424,426,429,448,449,451,454,473,474,476,479,498,499])
            elif origin == "top left":
                indexArray = np.array([8,17,23,33,42,49,51,52,73,74,76,77,83,91,106,115,120,123,133,142,148,151,152,158,
                167,183,192,198,201,202,223,224,226,227,233,242,251,258,267,283,291,298,301,302,308,317,326,327,333,342,351,372,
                374,375,383,392,399,401,402,423,424,426,421,448,449,458,467,473])
            elif origin == "top right":
                indexArray = np.array([3,9,18,24,28,34,43,49,53,59,68,74,96,97,98,99,103,109,118,124,128,134,143,149,153,159,
                168,174,178,184,193,199,208,209,210,223,228,234,243,249,253,259,268,274,296,297,298,299,303,309,318,324,346,
                347,348,349,371,372,373,374,396,397,398,399])
            elif origin == "light":
                indexArray = np.array([3,9,16,21,27,33,40,46,52,58,65,71,78,83,89,94,102,108,115,121,127,133,140,146,150,158,
                166,173,183,190,196,202,208,215,221,227,233,240,246,252,258,265,271,278,284,290,296,303,308,315,321,327,333,
                340,346,352,358,365,371,378,384,391,396])
            elif origin == "dark":
                indexArray = np.array([19,20,21,22,44,45,46,47,53,63,71,72,77,78,88,96,97,102,107,115,122,127,128,132,140,147,
                169,170,171,172,173,244,247,245,246])
            if i in indexArray:
                memoryCards.append(img)
            else:
                negCards.append(img)
    print("There are {} memory cards images in the dataset".format(len(memoryCards)))
    print("There are {} negative images in the dataset".format(len(negCards)))

    #labels of our data
    cardsLabels = np.ones(len(memoryCards))
    negLabels = np.zeros(len(negCards))

    #set the x and y set
    x = np.array(memoryCards + negCards, dtype='object')
    y = np.array(list(cardsLabels) + list(negLabels))
    print("Shape of image set ", x.shape, "Shape of labels ", y.shape)

    return x, y

def ML():
    folders = os.listdir('./cropped_imgs')
    i = 0
    for folder in folders:
        if os.path.isdir('./cropped_imgs/' + folder):
            x, y = loadAndPreprocess('./cropped_imgs/' + folder, folder)

            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
            # Creating a SVC object
            svc = SVC()

            # We'll use Cross Validation Grid Search to find best parameters.
            # Classifier will be trained using each parameter
            svc = SVC().fit(x_train,y_train)

            accuracy_train = accuracy_score(y_train, svc.predict(x_train))
            accuracy_test = accuracy_score(y_test, svc.predict(x_test))
            print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
            print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))
            print('\n')

            #detect and print on an image
            detectAndPrint(folder, svc, i)
            i +=1

def detectAndPrint(folder, svc, i):
    #load dictionnary and params
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    arucoParam = cv2.aruco.DetectorParameters_create()

    #load matrix and distortion coefficient from yaml file
    with open('calibration.yaml', 'r') as f:
        loadeddict = yaml.safe_load(f)
    mtx = np.array(loadeddict.get('camera_matrix'))
    dist_coeff = np.array(loadeddict.get('dist_coeff'))

    #load a random image
    image = None
    isImage = False
    listImg = os.listdir('./imgTest/' + folder)

    while (isImage == False):
        nb = random.randint(0, len(listImg)-1)
        image = listImg[nb]
        try:
            im=Image.open('./imgTest/' + folder + '/' + image)
            isImage = True
        except:
            isImage = False
    print('./imgTest/' + folder + '/' + image)
    image = cv2.imread('./imgTest/' + folder + '/' + image)
    
    #undistord it
    h,w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coeff, (w,h), 1, (w,h))
    undistorted_img = cv2.undistort(image, mtx, dist_coeff, None, newcameramtx)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    #find 2D and 2D of corners
    aruCo2D, aruCo3D = findCoordCorners(i, 'random', undistorted_img, arucoDict, arucoParam)

    #initialise 3D position of blocs that may contains Memory Cars
    blocs3D = get_blocs3D()

    #initialise 2D position of blocs that may contains Memory Cars
    blocs2D = get_blocs2D(i, blocs3D, aruCo2D, aruCo3D, mtx, dist_coeff, undistorted_img)
    blocs2D = np.array(blocs2D, dtype=np.int32)

    imageGray = cv2.cvtColor(undistorted_img.copy(), cv2.COLOR_BGR2GRAY)

    #cropped image, test if there is a memory on the cropped and add a rectangle if there is
    for index  in range(25):
        x1, y1 = np.array(blocs2D[index]).squeeze()
        x2, y2 = np.array(blocs2D[index+25]).squeeze()
        x3, y3 = np.array(blocs2D[index+50]).squeeze()
        x4, y4 = np.array(blocs2D[index+75]).squeeze()

        cnt = np.array([[[x1, y1]],[[x2, y2]],[[x3, y3]],[[x4, y4]]])

        # find the rotated rectangle enclosing the contour
        rect = cv2.minAreaRect(cnt)

        #cropped image
        extracted_image = crop_rect(imageGray, rect)

        # extracting cropped and predict the result with the model
        extracted_image = cv2.resize(extracted_image,(16,245)).ravel().reshape(1, -1)
        decision = svc.predict(extracted_image)
        
        if decision[0] == 1:
            #print a rectangle on the result image
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            undistorted_img = cv2.drawContours(undistorted_img,[box],0,(0,0,255),4)
            print("FOUND MEMORY CARD ON cropped number ", index)
    cv2.imwrite('./Images Result/' + folder + "_result.jpg", undistorted_img)

#luanch the renameing of all images in the folder imgTest
#renameImages("./imgTest")

#launch the detection of aruco, the projection of the 3D points and the saving of the generating cropped images
extractDatas("./imgTest")

#launch the machine learning model and test on a random image found in each folder in imgTest
#ML()


