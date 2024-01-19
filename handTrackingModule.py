import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, complexity=1, numHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.numHands = numHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.numHands, self.complexity,
                                        self.detectionCon, self.trackCon)  # is a function to find landmark
        self.mpDraw = mp.solutions.drawing_utils  # function to draw the lines on the hand

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo = 0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape                     #height, width, channel
                cx, cy = int(lm.x * w), int(lm.y * h)   #landmark x,y is converted in terms of height and width
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList







def main():
    pTime = 0
    cTime = 0
    cam = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cam.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




