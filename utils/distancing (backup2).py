import cv2
import itertools

class Person :
    def __init__(self, id, height, bottom):
        self.id = id
        self.height = height
        self.bottom = bottom
        self.isRed = 0

    def already_red(self):
        return self.isRed == 1


def show_distancing(img, peopleCoords):
    # Plot lines connecting people
    imgHeight, imgWidth, imgChannel = img.shape ##

    tl = round(0.001 * (imgHeight + imgWidth) / 2) + 1
    tf = max(tl - 1, 1)
    radius = 7
    thickness = 5
    colorBlue   = [255, 0, 0]
    colorYellow = [0, 255, 255]
    colorRed    = [0, 0, 255]
    colorGreen  = [0, 255, 0]

    peopleList = []

    for id, coord in enumerate(peopleCoords):
        pLeft, pTop, pRight, pBot = coord

        height = int(pBot - pTop) ## height
        bottom = int(pLeft + pRight) //2, int(pBot) ## get bottom center 1
        peopleList.append(Person(id, height, bottom))
        cv2.circle(img, bottom, radius, colorGreen, -1) ## bottom circle 2
        cv2.putText(img, str(id), (pLeft, pTop), 0, tl / 3, colorBlue, thickness=tf, lineType=cv2.LINE_AA) ##
        #cv2.putText(img, str(height), (pLeft, pTop), 0, tl / 3, colorBlue, thickness=tf, lineType=cv2.LINE_AA) ##

    peopleId = list(range(len(peopleCoords)))
    peopleIdCombi = list(itertools.combinations(peopleId, 2))

    for id1, id2 in peopleIdCombi:
        person1 = peopleList[id1]
        person2 = peopleList[id2]

        ## Ignore Perspective
        if abs(person1.height - person2.height) > 30:
            continue;

        ## Calculate with Image Ratio
        imgRatio = (person1.height + person2.height) / (170 + 170)
        distHighRisk = 200 * imgRatio
        distLowRisk = 250 * imgRatio

        botm1 = person1.bottom
        botm2 = person2.bottom
        dist = ((botm2[0]-botm1[0])**2 + (botm2[1]-botm1[1])**2)**0.5

        if dist < distHighRisk:
            person1.isRed = 1
            person2.isRed = 1
            cv2.line(img, botm1, botm2, colorRed, thickness)
            cv2.circle(img, botm1, radius, colorRed, -1)
            cv2.circle(img, botm2, radius, colorRed, -1)

        elif dist >= distHighRisk and dist < distLowRisk: 
            distCenter = (botm1[0] + botm2[0]) //2, (botm1[1] + botm2[1]) //2 ## get line center
            cv2.line(img, botm1, botm2, colorYellow, thickness)
            cv2.putText(img, str(int(dist)), distCenter, 0, tl / 3, colorYellow, thickness=tf, lineType=cv2.LINE_AA) ## put dist on line center
            if person1.already_red:
                cv2.circle(img, botm1, radius, colorYellow, -1)
            if person2.already_red:
                cv2.circle(img, botm2, radius, colorYellow, -1)

    return img


