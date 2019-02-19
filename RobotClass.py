__author__ = 'Caleytown'

class Robot:
    def __init__(self, xLoc, yLoc):
        self.xLoc = xLoc
        self.yLoc = yLoc

    def getLoc(self):
        return (self.xLoc,self.yLoc)

    def move(self, direction):
        if direction == 'left':
            self.xLoc = self.xLoc-1
        elif direction =='right':
            self.xLoc = self.xLoc+1
        elif direction == 'down':
            self.yLoc = self.yLoc + 1
        elif direction == 'up':
            self.yLoc = self.yLoc - 1
        else:
            self.xLoc = self.xLoc
            self.yLoc = self.yLoc

