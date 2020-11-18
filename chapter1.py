from cv2 import cv2
import numpy as np

#source env/bin/activate
print("Package Imported")

def editImage():
    img = cv2.imread("./lambo.jpeg")
    print(img.shape)
    imgResize = cv2.resize(img, (300, 200))
    cv2.imshow("Image", img)
    cv2.imshow("Image", imgResize)
    print(imgResize.shape)
    imgCropped = img[0:150, 100:150]
    cv2.imshow("Image", imgCropped)
    cv2.waitKey(0)

def useVideo():
    #cv2.imshow("Output", img )

    cap = cv2.VideoCapture("./video.mp4")

    cap = cv2.VideoCapture(0) 
    
    while(True): 
        
        # Capture the video frame 
        # by frame 
        ret, frame = cap.read() 
    
        # Display the resulting frame 
        cv2.imshow('frame', frame) 
        
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    cap.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 

def imgProcessing():

    kernel = np.ones((5, 5), np.uint8)

    img = cv2.imread("./lena.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)
    imgCanny = cv2.Canny(img, 100, 100)
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
    imgErosion = cv2.erode(imgDilation, kernel, iterations=1)

    cv2.imshow("Gray Image", imgGray)
    cv2.imshow("Blur Image", imgBlur)
    cv2.imshow("Canny Image", imgCanny)
    cv2.imshow("Dilation Image", imgDilation)
    cv2.imshow("Erosion Image", imgErosion)
    cv2.waitKey(0) 
 
def shapeDraw():

    img = np.zeros((512, 512, 3), np.uint8)
    # img[:] = 120,28, 67
    cv2.line(img, (0,0), (img.shape[1],img.shape[0]), (0, 255, 0), 3)
    cv2.rectangle(img, (0, 0), (250, 350),  (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (400, 50), 30, (255, 255, 0), 5)
    cv2.putText(img, " OPENCV ", (300, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 1)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def warpPerspective():
    img = cv2.imread("./cards.jpg")
    width,height = 250,350
    imgResize = cv2.resize(img, (550, 350))
    # pts1 = np.float32([[364, 83], [469, 113], [422, 256], [317, 222]])
    pts1 = np.float32([[364, 83], [317, 222], [422, 253], [469, 113]])
    pts2 = np.float32([[0,0], [0,height], [width,height], [width,0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(imgResize, matrix, (width,height))

    cv2.imshow("Image", imgResize)
    cv2.imshow("Output", imgOutput)
    #print(img.shape)
    cv2.waitKey(0)

def detectColor():

    def empty(a):
        pass


    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty )
    cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty )
    cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty )
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty )
    cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty )
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty )

    while True:
        img = cv2.imread("./lambo.jpeg")
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat  Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        #print(v_min)
        # lower1 = np.array([h_min, s_min, v_min])
        # upper1 = np.array([h_max, s_max, v_max])
        lower = np.array([000, 000, 186])
        upper = np.array([179, 255, 255])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask )

        cv2.imshow("Original", img)
        cv2.imshow("HSV", imgHSV)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", imgResult)
        cv2.waitKey(0)


detectColor()