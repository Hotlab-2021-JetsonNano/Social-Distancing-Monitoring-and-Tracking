import cv2
import itertools

class Colors:
    def __init__(self):
        self.blue = [255, 0, 0]
        self.yellow = [0, 255, 255]
        self.red = [0, 0, 255]
        self.green = [0, 255, 0]

class Configs:
    def __init__(self, img):
        self.colors = Colors()
        self.fontScale = round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
        self.fontThickness = max(self.fontScale - 1, 1)
        self.lineThickness = 2
        self.lineType = cv2.LINE_AA
        self.radius = 4
    
    def get_colors(self): 
        return self.colors

    def get_figure(self):
        return self.fontScale / 3, self.fontThickness, self.lineThickness, self.lineType, self.radius

class Person:
    def __init__(self, height, coord):
        self.height = height
        self.coord = coord
        self.redCount = 0
        self.updated = False

    def get_height(self):
        return self.height

    def set_height(self, height):
        self.height = height
        return

    def get_coord(self):
        return self.coord

    def set_coord(self, coord):
        self.coord = coord
        return

    def get_updated(self):
        return self.updated

    def set_updated(self, value):
        self.updated = value
        return

    def get_redCount(self):
        return self.redCount
    
    def inc_redCount(self):
        self.redCount += 1
        return

    def is_not_red(self):
        return self.redCount == 0

    def clear_redCount(self):
        self.redCount = 0
        return


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1)**2 + (y2 - y1)**2) **0.5
    
def is_valid_height_difference(height1, height2):
    return abs(height1 - height2) > 30

def calculate_distance_threshold(height1, height2):
    imgRatio = (height1 + height2) / (170 + 170)
    distHighRisk = 200 * imgRatio
    distLowRisk = 250 * imgRatio
    return distHighRisk, distLowRisk

def show_distancing(img, boxes, peopleList):
    config = Configs(img)
    color = config.get_colors()
    fontScale, fontThickness, lineThickness, lineType, radius = config.get_figure()

    isFirstFrame = (len(peopleList) == 0)

    for box in boxes:
        ## new data
        pLeft, pTop, pRight, pBot = box
        height = int(pBot - pTop)
        newCoord = (int(pLeft + pRight) //2, int(pBot))  # bottom

        if isFirstFrame:
            peopleList.append(Person(height, newCoord))

        ## update coord
        else :
            minDistance = 30
            minDistanceIdx = -1

            for idx, person in enumerate(peopleList):
                if person.get_updated() is True:
                    continue

                distance = calculate_distance(newCoord, person.get_coord())
                if minDistance > distance:
                    minDistance = distance
                    minDistanceIdx = idx
            
            if minDistanceIdx == -1:
                newPerson = Person(height, newCoord)
                newPerson.set_updated(True)
                peopleList.append(newPerson)
            else :
                peopleList[minDistanceIdx].set_height(height)
                peopleList[minDistanceIdx].set_coord(newCoord)
                peopleList[minDistanceIdx].set_updated(True)
        
        cv2.circle(img, newCoord, config.radius, color.green, -1) ## bottom circle 2
        #cv2.putText(img, str(id), (pLeft, pTop), 0, fontScale, colorBlue, fontThickness, lineType) ##
        #cv2.putText(img, str(height), (pLeft, pTop), 0, tl / 3, colorBlue, thickness=tf, lineType=cv2.LINE_AA) ##

    ## remove people not updated
    newPeopleList = []
    for person in peopleList:
        if person.get_updated() is True:
            person.set_updated(False)
            newPeopleList.append(person)
            cv2.putText(img, str(person.get_redCount()), person.get_coord(), 0, fontScale, color.red, fontThickness, lineType) 
    peopleList = newPeopleList

    ## make combinations of idx
    peopleIdx = list(range(len(peopleList))) ## 0, 1, 2, ..... , peopleList length
    peopleIdxCombi = list(itertools.combinations(peopleIdx, 2))

    for idx1, idx2 in peopleIdxCombi:
        person1 = peopleList[idx1]
        person2 = peopleList[idx2]

        ## Ignore Perspective
        if is_valid_height_difference(person1.get_height(), person2.get_height()):
            continue

        ## get distance info
        dist = calculate_distance(person1.get_coord(), person2.get_coord())
        distHighRisk, distLowRisk = calculate_distance_threshold(person1.get_height(), person2.get_height())  ## Calculate with Image Ratio

        ## high risk
        if dist < distHighRisk:
            if person1.get_updated() is False:
                person1.inc_redCount()
                person1.set_updated(True)
            if person2.get_updated() is False:
                person2.inc_redCount()
                person2.set_updated(True)                    
            
            cv2.line(img, person1.get_coord(), person2.get_coord(), color.red, lineThickness)
            cv2.circle(img, person1.get_coord(), radius, color.red, -1)
            cv2.circle(img, person2.get_coord(), radius, color.red, -1)

        ## low risk
        elif dist >= distHighRisk and dist < distLowRisk:
            if person1.get_updated() is False:
                person1.clear_redCount()
            if person2.get_updated() is False:
                person2.clear_redCount()
            
            cv2.line(img, person1.get_coord(), person2.get_coord(), color.yellow, lineThickness)
            if person1.is_not_red():
                cv2.circle(img, person1.get_coord(), radius, color.yellow, -1)
            if person2.is_not_red():
                cv2.circle(img, person2.get_coord(), radius, color.yellow, -1)

    for idx in peopleIdx:
        if peopleList[idx].get_updated() is False:
            peopleList[idx].clear_redCount()
        else :
            peopleList[idx].set_updated(False)

    return img
