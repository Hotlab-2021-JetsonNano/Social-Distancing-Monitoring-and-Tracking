import cv2
import time

class Colors:
    def __init__(self):
        self.blue   = (255, 0, 0)
        self.yellow = (0, 255, 255)
        self.red    = (0, 0, 255)
        self.green  = (0, 255, 0)
        self.black  = (0, 0, 0)

class Configs:
    def __init__(self, img):
        self.colors = Colors()
        self.imgRatio = img.shape[1] / 720
        self.fontScale = round(1 * self.imgRatio)
        self.fontThickness = round(1 * self.imgRatio)
        self.lineThickness = 2
        self.lineType = cv2.LINE_AA
        self.radius = round(3 * self.imgRatio)
    
    def get_colors(self): 
        return self.colors

    def get_figure(self):
        return self.imgRatio, self.fontScale / 3, self.fontThickness, self.lineType, self.radius

    def get_figure_line(self):
        return self.lineThickness, self.lineType
    

class Person:
    def __init__(self, id, height, coord):
        self.id = id
        self.height = height
        self.coord = coord
        self.redCount = 0
        self.riskTime = 0.0
        self.isUpdated = False
        self.isYellow = False
        self.missCount = 0

    def reset(self, height, coord):
        self.height = height
        self.coord = coord
        self.isUpdated = False
        self.isYellow = False
        self.missCount = 0
        return

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id
        return

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

    def is_updated(self):
        return self.isUpdated
    def set_updated(self, value):
        self.isUpdated = value
        return

    def is_yellow(self):
        return self.isYellow
    def set_yellow(self, value):
        self.isYellow = value
        return

    def get_redCount(self):
        return self.redCount
    def inc_redCount(self):
        self.redCount += 1
        return
    def clear_redCount(self):
        self.redCount = 0
        return
    # def is_definite_risk(self):
    #     return self.redCount > 30

    def get_riskTime(self):
        return self.riskTime
    def inc_riskTime(self, time):
        self.riskTime = round(self.riskTime + time, 2)
        return
    def clear_riskTime(self):
        self.riskTime = 0.0
        return
    def is_definite_risk(self):
        return self.riskTime > 10.0

    def is_erasable(self, thres):
        x, y = self.coord
        low_x, low_y = (3, 3)
        high_y, high_x = thres
        return (x < low_x or x > high_x - 3  or y < low_y or y > high_y - 3)

    def inc_missCount(self):
        self.missCount += 1
        return
    def clear_missCount(self):
        self.missCount = 0
        return
    def is_missable(self):
        return self.missCount > 20

class IdTable:
    def __init__(self):
        self.peopleList = []
        self.personIdList = []
        self.parentIdList = []
        self.groupCoordsList = []
        self.peopleRisk = set([])

    def init_idList(self, invalidIdList):
        self.personIdList = [person.get_id() for person in self.peopleList]
        for index, personId in enumerate(self.personIdList):
            if personId == -1:
                newId = max(self.personIdList) + 1
                while newId in invalidIdList:
                    newId += 1
                self.peopleList[index].set_id(newId)
                self.personIdList[index] = newId
        self.parentIdList = self.personIdList[:]
        return

    def init_groupList(self):
        self.groupCoordsList = [[] for i in range(len(self.peopleList))]
        return

    def get_people(self):
        return self.peopleList

    def get_person(self, index):
        return self.peopleList[index]

    def add_person(self, person):
        self.peopleList.append(person)
        return

    def get_ids(self, person):
        index = self.personIdList.index(person.get_id())
        return self.personIdList[index], self.parentIdList[index]

    def get_personIdx(self, personId):
        return self.personIdList.index(personId)

    def get_parentIdx(self, parentId):
        return self.parentIdList.index(parentId)

    def set_parentId(self, personId, parentId):
        index = self.personIdList.index(personId)
        self.parentIdList[index] = parentId
        return

    def find_parentId(self, personId):
        index    = self.personIdList.index(personId)
        parentId = self.parentIdList[index]
        if personId == parentId:
            return parentId
        else :
            #personId = parentId
            parentId = self.find_parentId(parentId)
            #self.set_parentId(personId, parentId)
            return parentId

    def merge_parentIds(self, personId1, personId2):
        parentId1 = self.find_parentId(personId1)
        parentId2 = self.find_parentId(personId2)
        self.set_parentId(parentId1, parentId2)
        return

    def get_groupList(self):
        return self.groupCoordsList

    def set_groupList(self, personId, personCoord):
        parentId = self.find_parentId(personId)
        parentIdx = self.get_parentIdx(parentId)
        self.groupCoordsList[parentIdx].append(personCoord)
        return

    def update_red(self, index, frameTime):
        if self.peopleList[index].is_updated():
            self.peopleList[index].inc_riskTime(frameTime)
            self.peopleList[index].set_updated(True)
        return

    def update_yellow(self, index):
        self.peopleList[index].set_yellow(True)
        return
        

class FrameData:
    def __init__(self):
        self.peopleList = []
        self.validIdList   = set([])
        self.invalidIdList = set([])
        self.counter = 1
        self.fps = 0.0
        self.tic = 0.0
        self.toc = 0.0
        self.log = ""

    def get_people_len(self):
        return len(self.peopleList)

    def get_people(self):
        return self.peopleList

    def set_people(self, peopleList):
        self.peopleList = peopleList[:]
        return

    def poll_person(self, index):
        person = self.peopleList[index]
        del self.peopleList[index]
        return person

    def init_invalid(self):
        self.invalidIdList = set([person.get_id() for person in self.peopleList])
        return

    def get_invalid(self):
        return self.invalidIdList
    
    def get_valid_len(self):
        return len(self.validIdList)

    def get_valid_min(self):
        id = min(self.validIdList)
        self.validIdList.remove(id)
        return id

    def set_valid(self, id):
        self.invalidIdList.remove(id)
        self.validIdList.add(id)
        return

    def get_counter(self):
        return self.counter

    def increase_counter(self):
        self.counter += 1
        return

    def set_timer(self):
        self.tic = time.time()
        return

    def get_fps(self):
        return self.fps
    
    def update_fps(self):
        self.toc = time.time()
        curr_fps = 1.0 / (self.toc - self.tic)
        self.fps = curr_fps if self.fps == 0.0 else (self.fps*0.95 + curr_fps*0.05)
        self.tic = self.toc
        return

    def get_log(self):
        return self.log

    def update_log(self, log):
        self.log += log
        return
    
    def clear_log(self):
        self.log = ""
        return

    ## display 
    def show_fps(img, fps):
        """Draw fps number at top-left corner of the image."""
        font = cv2.FONT_HERSHEY_PLAIN
        line = cv2.LINE_AA
        fps_text = 'FPS: {:.2f}'.format(fps)
        cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
        cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
        return img
