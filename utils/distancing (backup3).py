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
        self.lineThickness = 5
        self.lineType = cv2.LINE_AA
        self.radius = 7
    
    def get_colors(self): 
        return self.colors

    def get_figure(self):
        return self.fontScale / 3, self.fontThickness, self.lineThickness, self.lineType, self.radius

class Person:
    def __init__(self, id, height, bottom):
        self.id = id
        self.height = height
        self.bottom = bottom
        self.redCount = 0

    def get_id(self):
        return self.id

    def get_height(self):
        return self.height

    def get_coords(self):
        return self.bottom

    def is_not_red(self):
        return self.redCount == 0
    
    def inc_redCount(self):
        self.redCount += 1

class People:
    def __init__(self):
        self.peopleHighRisk = []
        self.isFirstFrame = True
    
    def get_isFirstFrame(self):
        return self.isFirstFrame

    def set_isFirstFrame(self, value):
        self.isFirstFrame = value
        return

    def get_length(self):
        return self.peopleHighRisk.__len__()

    def update_person(self, i, coord):
        self.peopleHighRisk[i].set_coord(coord)
        self.peopleHighRisk[i].inc_redCount()
        return

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1)**2 + (y2 - y1)**2) **0.5
    
def calculate_height_difference(height1, height2):
    return abs(height1 - height2) > 30

def calculate_distance_threshold(height1, height2):
    imgRatio = (height1 + height2) / (170 + 170)
    distHighRisk = 200 * imgRatio
    distLowRisk = 250 * imgRatio
    return distHighRisk, distLowRisk

def show_distancing(img, coords, people):
    config = Configs(img)
    colorBlue, colorYellow, colorRed, colorGreen = config.get_colors()
    fontScale, fontThickness, lineThickness, lineType, radius = config.get_figure()

    peopleList = []

    for id, coord in enumerate(coords):
        pLeft, pTop, pRight, pBot = coord

        height = int(pBot - pTop) ## height
        bottom = int(pLeft + pRight) //2, int(pBot) ## get bottom center 1
        peopleList.append(Person(id, height, bottom)) 
        
        cv2.circle(img, bottom, config.radius, colorGreen, -1) ## bottom circle 2
        #cv2.putText(img, str(id), (pLeft, pTop), 0, fontScale, colorBlue, fontThickness, lineType) ##
        #cv2.putText(img, str(height), (pLeft, pTop), 0, tl / 3, colorBlue, thickness=tf, lineType=cv2.LINE_AA) ##

    peopleId = list(range(len(coords)))
    peopleIdCombi = list(itertools.combinations(peopleId, 2))

    for id1, id2 in peopleIdCombi:
        person1 = peopleList[id1]
        person2 = peopleList[id2]

        ## Ignore Perspective
        if calculate_height_difference(person1.get_height(), person2.get_height()):
            continue

        ## get distance info
        dist = calculate_distance(person1.get_bottom(), person2.get_bottom())
        distHighRisk, distLowRisk = calculate_distance_threshold(person1.get_height(), person2.get_height())  ## Calculate with Image Ratio

        ## compare distance
        # high risk
        if dist < distHighRisk:
            ## Tracking
            if people.get_isFirstFrame():
                if person1.get_id() in peopleId:
                    person1.inc_redCount()
                    people.peopleHighRisk.append(person1)
                    peopleId.remove(person1.get_id())
                if person2.get_id() in peopleId:
                    person2.inc_redCount()
                    people.peopleHighRisk.append(person2)
                    peopleId.remove(person2.get_id())
            else :
                minDistance1, minId1 = 100000, 0
                minDistance2, minId2 = 100000, 0
                for idx, person in enumerate(people.peopleHighRisk):
                    distance = calculate_distance(person.get_coord(), person1.get_coord())
                    if distance < 50 and distance < minDistnace:
                        minDistnace = distance
                        return True
                    else :
                        return False
                    
                    # if is_adjacent_coord(people.peopleHighRisk[i].get_coords, person1.get_coords, minDistance1, minId1): ## 인접한 픽셀이다 and 최소 거리에 있다
                    #     people.peopleHighRisk.update_person(i, person1.get_coord)
                    # else :
                    #     person1.inc_redCount()
                    #     people.peopleHighRisk.append(person1)
                    # if is_adjacent_coord(people.peopleHighRisk[i].get_coords, person2.get_coords, minDistance2, minId2): ## 인접한 픽셀이다 and 최소 거리에 있다
                    #     people.peopleHighRisk.update_person(i, person2.get_coord)
                    # else :
                    #     person2.inc_redCount()
                    #     people.peopleHighRisk.append(person2)

                    ## update 안 된 사람은 제거해야 한다.
                    ## redCount가 몇 프레임 이상 증가하면 로그를 남겨야 한다.
                    
            cv2.line(img, (x1, y1), (x2, y2), colorRed, lineThickness)
            cv2.circle(img, (x1, y1), radius, colorRed, -1)
            cv2.circle(img, (x2, y2), radius, colorRed, -1)

        # low risk
        elif dist >= distHighRisk and dist < distLowRisk: 
            distCenter = (x1 + x2) //2, (y1 + y2) //2 ## get line center
            cv2.line(img, (x1, y1), (x2, y2), colorYellow, lineThickness)
            cv2.putText(img, str(int(dist)), distCenter, 0, fontScale, colorYellow, fontThickness, lineType) ## put dist on line center
            if person1.is_not_red():
                cv2.circle(img, (x1, y1), radius, colorYellow, -1)
            if person2.is_not_red():
                cv2.circle(img, (x2, y2), radius, colorYellow, -1)

    if people.get_isFirstFrame():
        people.set_isFirstFrame(False)

    return img


