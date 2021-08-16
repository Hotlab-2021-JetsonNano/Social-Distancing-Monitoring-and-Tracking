import cv2
import itertools

class Person :
    def __init__(self, id, bottom):
        self.id = id
        self.bottom = bottom
        self.isRed = 0

    def get_bottom(self):
        return self.bottom

    def check_red(self):
        return self.isRed == 1

def distancing(peopleCoords, img):
    # Plot lines connecting people
    imgHeight, imgWidth, imgChannel = img.shape ##
    
    already_red = dict() # dictionary to store if a plotted rectangle has already been labelled as high risk
    bottomList = [] ## bottom list
    peopleList = []

    colorBlue   = [255, 0, 0]
    colorYellow = [0, 255, 255]
    colorRed    = [0, 0, 255]
    colorGreen  = [0, 255, 0]
    tl = round(0.001 * (imgHeight + imgWidth) / 2) + 1
    tf = max(tl - 1, 1)
    radius = 7
    thickness = 5

    for coord in peopleCoords:
        pLeft, pTop, pRight, pBot = int(coord);

        bottom = (pLeft + pRight) //2, pBot ## get bottom center 1
        bottomList.append(bottom) 
        already_red[bottom] = 0
        peopleList.append(Person(id, bottom))

        height = str(pBot - pTop) ## height
        coord = pLeft, pTop
        cv2.putText(img, height, coord, 0, tl / 3, colorBlue, thickness=tf, lineType=cv2.LINE_AA) ##

    peopleCombi = list(itertools.combinations(peopleCoords, 2))

    for combi in peopleCombi:
        pLeft1, pTop1, pRight1, pBot1 = int(combi[0]);
        pLeft2, pTop2, pRight2, pBot2 = int(combi[1]);

        height1 = pBot1 - pTop1
        height2 = pBot2 - pTop2  

        if abs(height1 - height2) > 30:
            continue;

        botm1 = (pRight1 + pLeft1) //2, pBot1 ## get bottom center 1
        botm2 = (pRight2 + pLeft1) //2, pBot1 ## get bottom center 2
        dist = ((botm2[0]-botm1[0])**2 + (botm2[1]-botm1[1])**2)**0.5

        distRate = (height1 + height2) / (170 + 170) ##
        dist_low = 200 * distRate ##
        dist_high = 250 * distRate ##

        if dist < dist_low:
            already_red[botm1] = 1
            already_red[botm2] = 1
            cv2.line(img, botm1, botm2, colorRed, thickness)
            cv2.circle(img, botm1, radius, colorRed, -1)
            cv2.circle(img, botm2, radius, colorRed, -1)

        elif dist > dist_low and dist < dist_high: 
            distCenter = (int((botm1[0] + botm2[0]) / 2), int((botm1[1] + botm2[1]) / 2)) ## get line center
            cv2.line(img, botm1, botm2, colorYellow, thickness)
            cv2.putText(img, str(int(dist)), distCenter, 0, tl / 3, colorYellow, thickness=tf, lineType=cv2.LINE_AA) ## put dist on line center
            if already_red[botm1] == 0:
                cv2.circle(img, botm1, radius, colorYellow, -1)
            if already_red[botm2] == 0:
                cv2.circle(img, botm2, radius, colorYellow, -1)

        else :
            if already_red[botm1] == 0:
                cv2.circle(img, botm1, 10, colorGreen, -1) ## bottom circle 2
            if already_red[botm2] == 0:
                cv2.circle(img, botm2, 10, colorGreen, -1) ## bottom circle 2
