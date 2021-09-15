from queue import Queue

class ThreadQueue:
    def __init__(self):
        self.possible = True
        self.imgQueue = Queue(1)
        self.trtQueue = Queue(1)
        self.letQueue = Queue(1)

    def waitMainThread(self):
        #self.check.join()
        return
    
    def putThreadQueue(self, img, trt_outputs, letter_box):
        self.imgQueue.put(img)
        self.trtQueue.put(trt_outputs)
        self.letQueue.put(letter_box)
        return

    def getThreadQueue(self):
        if self.isImpossible() and self.isEmpty():
            self.destroy()
        return self.imgQueue.get(), self.trtQueue.get(), self.letQueue.get()

    def signalMainThread(self):
        self.imgQueue.task_done()
        return

    def setImpossible(self):
        self.possible = False
        return

    def isImpossible(self):
        return self.possible != True

    def isPossible(self):
        return self.possible

    def isEmpty(self):
        return self.imgQueue.empty()        

    def destroy(self):
        del imgQueue
        del trtQueue
        del letQueue
        return