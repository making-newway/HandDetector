'''
Created on 04-May-2021

@author: SAYAN DEY
'''

import cv2
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.tips = [4, 8, 12, 16, 20]
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        
        self.mpDraw = mp.solutions.drawing_utils
        
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    
        return img
    
    
    def findPositions(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for iD, lm in enumerate(myHand.landmark):
                
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([iD, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            xmin, xmax, ymin, ymax = min(xList), max(xList), min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            
            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)
        
        return self.lmList, bbox
    
    def fingers(self):
        fingers = []
        
        if self.lmList[self.tips[0]][1] > self.lmList[self.tips[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for iD in range(1, 5):
            if self.lmList[self.tips[iD]][2] < self.lmList[self.tips[iD]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
    def findDistance(self, p1, p2, img, draw=True):
        
        x1, y1, x2, y2 = self.lmList[p1][1], self.lmList[p1][2], self.lmList[p2][1], self.lmList[p2][2]
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 200, 25), 3)
            
        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1, y1, x2, y2]
    
    
def main():
    pTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        hands = detector.findHands(img)
        lmList = detector.findPositions(img)
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (255, 0, 255), 3)
        
        cv2.imshow("Capture", img)
        cv2.waitKey(1)
        

if __name__ == "__main__":
    main()













        
        