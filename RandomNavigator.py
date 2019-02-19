__author__ = 'Caleytown'
import numpy as np
from random import randint

class RandomNavigator:
    #def __init__(self):
      #def getAction(self,robot, currentPos, movement, map):
    def getAction(self, currentPos, movement):

        #randNumb = randint(0, 3)
        if (currentPos[0] + movement[0] < 0) or (currentPos[0] + movement[0] > 27) or (currentPos[1] + movement[1] < 0) or (currentPos[1] + movement[1] > 27):
            #print('null')
            return 'null'
        elif (movement[0] == -1):
            #print('left')
            return 'left'
        elif (movement[0] == 1):
            #print('right')
            return 'right'
        elif (movement[1] == -1):
            #print('up')
            return 'up'
        elif (movement[1] == 1):
            #print('down')
            return 'down'

        '''
        randNumb = randint(0,3)
        if randNumb == 0:
            if robot.getLoc()[0]-1 < 0:
                randNumb = randNumb + 1
            else:
                return 'left'
        if randNumb == 1:
            if robot.getLoc()[0]+1 > 27:
                randNumb = randNumb + 1
            else:
                return 'right'
        if randNumb == 2:
            if robot.getLoc()[1]+1 > 27:
                randNumb = randNumb + 1
            else:
                return 'down'
        if randNumb == 3:
            if robot.getLoc()[1]-1 < 0:
                randNumb = 0
            else:
                return 'up'
        '''