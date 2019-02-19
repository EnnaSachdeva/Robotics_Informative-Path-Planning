__author__ = 'Caleytown'

'''
World is constructed from MNIST dataset, if the MNIST digit is 
0-2: goal is (0,27)
3-5: goal is (27, 27)
6-9: goal is (27, 0)
'''
import gzip
import numpy as np
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from RandomNavigator import RandomNavigator
from networkFolder.functionList import Map,WorldEstimatingNetwork,DigitClassifcationNetwork
import time


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw
import random
from numpy import unravel_index

from Astar import Node
'''
WorldEstimatingNetwork: Given what you have currently observed, this network tells you what the world looks like, 
masks the areas already explored.

DigitClassificationNetwork: This network takes this estimate of the world and provides the estimate of what digit 
the world belongs. Returns an array of size 10, with each index representing the different digits the world could be.

Robot has 2 features:

get_loc: gives the location of the robot
move:  moves the robot in either of the 4 directions

RandomNavigator: only 1 function
getAction: randomly gets action


Game has 2 characteristics:
tick: takes random action in either of 4 directions, moves the robot, updates the map,
and returns True if the goal is reached.

update_map: masks the location of the grid visited as already explored.
'''

'''
def goal_directed(currentLoc, pos, reward, steps, mask, goalLoc, wrongGoals):
    reached = False
    wrong = True
    while (currentLoc[0] != pos[0]):
        steps = steps+1
        mov_x = pos[0] - currentLoc[0] # cells along x axis
        direction = mov_x/abs(mov_x)
        game.tick(currentLoc, [direction, 0], 0)

        if (robot.getLoc()[0] == goalLoc[0]) and (robot.getLoc()[1] == goalLoc[1]):
            reached = True
            break


        # check if it accidently moved to wrong goal
        for m in range(len(wrongGoals)):
            if (robot.getLoc()[0] == wrongGoals[m][0]) and (robot.getLoc()[1] == wrongGoals[m][1]):
                wrong = True
                break

        if (wrong == True):
            break

        # Reward structure
        reward = reward - 1

        currentLoc = robot.getLoc()
        mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
        plt.plot(robot.xLoc, robot.yLoc, 'ro--', linewidth=2, markersize=15)  # current location of robot
        draw()

    if (reached == True) or (wrong == True): # if reached any of the goals
       break


    while (currentLoc[1] != pos[1]):
        steps = steps+1
        mov_y = pos[1] - currentLoc[1] # cells along x axis
        direction = mov_y/abs(mov_y)
        game.tick(currentLoc, [0, direction], 0)


        if (robot.getLoc()[0] == goalLoc[0]) and (robot.getLoc()[1] == goalLoc[1]):
            reached = True
            break


        # check if it accidently moved to wrong goal
        for m in range(len(wrongGoals)):
            if (robot.getLoc()[0] == wrongGoals[m][0]) and (robot.getLoc()[1] == wrongGoals[m][1]):
                wrong = True
                break

        if (wrong == True):
            break

        # Reward structure
        reward = reward - 1
        currentLoc = robot.getLoc()
        mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
        plt.plot(robot.xLoc, robot.yLoc, 'ro--', linewidth=2, markersize=15)  # current location of robot
        draw()

    if (reached == True) or (wrong == True):  # if reached any of the goals
        break

    return steps, reward, mask, reached, wrong
'''



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

################## Generate for all the classes ######################
rewards = []
for k in range(0, 10):

    map = Map(k)
    #map = MapgetNewMap()
    # Maximum information corresponds to 255 and min to 0
    data = map.map  # 28*28 map

    print("Map represents:", map.number) # number represented by the map

    # plot the map in the grid, using heatmap
    plt.imshow(data, cmap='hot', interpolation='nearest')
    #plt.ion()
    #plt.show()

    # Initialize the position of robot

    robot = Robot(0, 0)
    temprobot = Robot(0, 0)

    navigator = RandomNavigator()


    # Goal Location
    if (0<=map.number <=2):
        goalLoc = (0, 27)
        wrongGoals = [[27,27], [27,0]]

    elif (3<=map.number <=5):
        goalLoc = (27, 27)
        wrongGoals = [[0,27], [27,0]]

    elif (6<=map.number <=9):
        goalLoc = (27, 0)
        wrongGoals = [[0,27], [27,27]]


    game = Game(data, goalLoc, navigator, robot, temprobot) # this takes one action in the world and then updates the explored area of the map, also tell if the goal is reached or not

    ''' Robot looks at one step ahead, accordingly map gets explored and masked in those corresponding positions, 
        gets the prediction of the masked and explored map from the Neural network and then goes in the direction 
        which gives the maximum information, among all valid movements. The pixels in the next step can be in 
        between 0 or 255.   
    '''

    uNet = WorldEstimatingNetwork()
    classNet = DigitClassifcationNetwork()

    mask = np.zeros((28, 28))
    tempmask = np.zeros((28, 28))

    # intialize movement to update the mask and exploredMap
    game.tick(robot.getLoc(), [0, 0], 0)
    mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
    tempexploredMap = np.ones(data.shape) * 500

    steps = 0
    reward = 0
    for i in range(1000): # movements till it reaches the goal
        #time.sleep(0.1)
        currentLoc = [robot.getLoc()[0], robot.getLoc()[1]]


        #tempexploredMap = game.exploredMap

        image = uNet.runNetwork(game.exploredMap, mask)



        # Normalizing the image pixel values

        for x in range(28):
           for y in range(28):
               if (image[x, y] <= 0):
                   image[x, y] = 0


        char = classNet.runNetwork(image)  # what digit world belongs to: gives value corresponding to each class
        char = np.reshape(char, 10)

        # Normalizing the softmax values
        #char = [char[p] - np.min(char) for p in range(len(char))]
        #char = char / np.sum(char)
        char = softmax(char)
        # plot robot location in grid
        #plt.imshow(robot.xLoc, robot.yLoc)

        reached = False
        explored = False
        print("Current Position of robot", [robot.xLoc, robot.yLoc], "predicted character: ", char.argmax(), "with prob:", char.max(), "Pixel Value:", image[robot.xLoc, robot.yLoc])

        if (char[map.number] > 0.75): # if sufficient exploration happened
            explored = True
            print("explored")
            break

        plt.plot(robot.xLoc, robot.yLoc,  marker="s", color="c", linewidth=2, markersize=10) # current location of robot
        draw()
        #plt.ion()
        #plt.show()

        # robot looks one step ahead
        info = []

        temp = 0

        # move greedily towards white pixels
        found = 0



        tempImage = np.zeros(np.shape(image))
        #currentInfo = 100
        list = []
        tempList = []

        pixels = []
        tempPixels = []
        lookahead = 1
        currentInfo = char
        finished = False
        # immediate neighbors maximum pixels
        for x in range(-1, 2):
            for y in range(-1, 2):
                # check if the move is valid
                if (robot.getLoc()[0] + x < 0) or (robot.getLoc()[0] + x > 27) or (robot.getLoc()[1] + y < 0) \
                        or (robot.getLoc()[1] + y > 27) or (abs(x) == abs(y) != 0)\
                        or (game.exploredMap[robot.getLoc()[0] + x, robot.getLoc()[1] + y] != 500):
                    continue
                else:  # assume take action

                    if (x==0 and y==0):
                        currentInfo = char
                    else:
                        temp = 1
                        game.tick(robot.getLoc(), [x, y], 1) # assuming temporary movement
                        tempmask[robot.getLoc()[0] + x, robot.getLoc()[1] + y] = 1

                        ### this pixel can correspond to 0 or 255 or any value in between. Lets check for 0 and 255
                        check = [0, 255]
                        diff = []
                        for k in range(len(check)):
                            tempexploredMap[robot.getLoc()[0] + x, robot.getLoc()[1] + y] = check[k]
                            tempimage = uNet.runNetwork(tempexploredMap, tempmask)
                            # Normalize the image
                            #for row in range(28):
                            #    for col in range(28):
                            #        if (image[row, col] <= 0):
                            #            image[row, col] = 0

                            char = classNet.runNetwork(tempimage)
                            char = np.reshape(char, 10)
                            char = softmax(char)
                            #info.append(np.sum(abs(np.subtract(currentInfo, char))))
                            #info.append(char[map.number])
                            #info.append(np.max(np.subtract(currentInfo, char)))
                            diff.append(char)

                        info.append(np.sum(abs(np.subtract(diff[1], diff[0]))))
                        list.append([robot.getLoc()[0] + x, robot.getLoc()[1] + y])


        while (len(info)!=0):
            index = np.argmax(info)
            pos = list[index]
            if (game.exploredMap[pos[0], pos[1]] == 500) :# which means if that pos has been visited before, take the second best
                break
            else:
                info.remove(info[index])
                list.remove(list[index])






        #else:
        #    finished = True
        #    print("No more nearby pixels left to be found")
        #    break

        currentLoc = robot.getLoc()



        noPath = False
        while((currentLoc[0]!=pos[0]) or (currentLoc[1]!=pos[1])) and (noPath == False):
            #time.sleep(0.01)
            prev_x = currentLoc[0]
            prev_y = currentLoc[1]
            if(currentLoc[0] != pos[0]):
                steps = steps + 1
                mov_x = pos[0] - currentLoc[0]  # cells along x axis
                direction = np.int(mov_x / abs(mov_x)) # direction of movement
                prev_x = currentLoc[0]
                #if (mask[currentLoc[0]+direction][currentLoc[1]] != 1): # move only if the path has not been traversed before'

                game.tick(currentLoc, [direction, 0], 0)  # move along that direction
                plt.plot(robot.xLoc, robot.yLoc, marker="s", color="m", linewidth=2,
                        markersize=10)  # current location of robot
                #mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
                reward = reward-1
                currentLoc = robot.getLoc()

            if (currentLoc[1] != pos[1]):
                steps = steps + 1
                mov_y = pos[1] - currentLoc[1]
                direction = np.int(mov_y / abs(mov_y)) # direction of movement
                prev_y = currentLoc[1]
                #if (mask[currentLoc[0]][currentLoc[1]+ direction] != 1): # move only if the path has not been traversed before
                game.tick(currentLoc, [0, direction], 0)  # move along that direction
                plt.plot(robot.xLoc, robot.yLoc, marker="s", color="m", linewidth=2,
                         markersize=10)  # current location of robot
                #mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
                reward = reward - 1
                currentLoc = robot.getLoc()
                # direction_y = mov_y / abs(mov_y)

            if(prev_x == currentLoc[0]) and (prev_y == currentLoc[1]):
                noPath = True # no path exist

        if (noPath== True):
            break

        mask[robot.getLoc()[0], robot.getLoc()[1]] = 1


        # finding shortest distance towards this goal

        #pos = unravel_index(tempImage.argmax(), image.shape)

        # perform a temporary goal directed operation

        #currentLoc = robot.getLoc()

    ############## Reached either correct goal or wrong Goal #############

    print("Exploration ended at:", robot.getLoc())

    tempexploredMap = game.exploredMap

    # if exploration is done and the robot now needs to go to the correct location
    if (explored == True): # exploration done
        print("Exploration done, now move to actual goal")

        while (currentLoc[0]!=goalLoc[0]) or (currentLoc[1]!=goalLoc[1]):

            if (currentLoc[0] != goalLoc[0]):
                steps = steps + 1
                mov_x = goalLoc[0] - currentLoc[0]  # cells along x axis
                direction = np.int(mov_x / abs(mov_x))  # direction of movement
                info = []
                list = []
                for y in range(-1, 2):
                        # check if the move is valid
                        if (robot.getLoc()[0] + direction < 0) or (robot.getLoc()[0] + direction > 27) or (robot.getLoc()[1] + y < 0) \
                                or (robot.getLoc()[1] + y > 27) or (abs(direction) == abs(y) != 0):  # \
                            # or (game.exploredMap[robot.getLoc()[0] + x, robot.getLoc()[1] + y] != 500):
                            continue
                        else:  # assume take action

                            if (direction == 0 and y == 0):
                                currentInfo = char
                            else:
                                temp = 1
                                game.tick(robot.getLoc(), [direction, y], 1)  # assuming temporary movement
                                tempmask[robot.getLoc()[0] + direction, robot.getLoc()[1] + y] = 1

                                ### this pixel can correspond to 0 or 255 or any value in between. Lets check for 0 and 255
                                check = [0, 255]
                                diff = []
                                for k in range(len(check)):
                                    tempexploredMap[robot.getLoc()[0] + direction, robot.getLoc()[1] + y] = check[k]
                                    tempimage = uNet.runNetwork(tempexploredMap, tempmask)
                                    # Normalize the image
                                    # for row in range(28):
                                    #    for col in range(28):
                                    #        if (image[row, col] <= 0):
                                    #            image[row, col] = 0

                                    char = classNet.runNetwork(tempimage)
                                    char = np.reshape(char, 10)
                                    char = softmax(char)
                                    # info.append(np.sum(abs(np.subtract(currentInfo, char))))
                                    # info.append(char[map.number])
                                    # info.append(np.max(np.subtract(currentInfo, char)))
                                    diff.append(char)

                                info.append(np.sum(abs(np.subtract(diff[1], diff[0]))))
                                list.append([robot.getLoc()[0] + direction, robot.getLoc()[1] + y])

                while (len(info) != 0):
                    index = np.argmax(info)
                    pos = list[index]
                    if (game.exploredMap[pos[0], pos[1]] == 500):  # which means if that pos has not been visited before, take the second best
                        break
                    else:
                        info.remove(info[index])
                        list.remove(list[index])

                game.tick(currentLoc, [pos[0]-currentLoc[0], pos[1]-currentLoc[1]], 0)  # move along that direction
                plt.plot(robot.xLoc, robot.yLoc, marker="s", color="m", linewidth=2,
                        markersize=10)  # current location of robot
                mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
                reward = reward - 1
                currentLoc = robot.getLoc()

                image = uNet.runNetwork(game.exploredMap, mask)

                char = classNet.runNetwork(image)  # what digit world belongs to: gives value corresponding to each class
                char = np.reshape(char, 10)

                char = softmax(char)
                print("Current Position of robot", [robot.xLoc, robot.yLoc], "predicted character:", char.argmax(),
                      "with prob:", char.max(), "Pixel Value:", image[robot.xLoc, robot.yLoc])


            if (currentLoc[1] != goalLoc[1]):
                steps = steps + 1
                mov_y = goalLoc[1] - currentLoc[1]
                direction = np.int(mov_y / abs(mov_y))  # direction of movement

                info = []
                list = []
                for x in range(-1, 2):
                    # check if the move is valid
                    if (robot.getLoc()[0] + x < 0) or (robot.getLoc()[0] + x > 27) or (
                            robot.getLoc()[1] + direction < 0) \
                            or (robot.getLoc()[1] + direction > 27) or (abs(x) == abs(direction) != 0):  # \
                        # or (game.exploredMap[robot.getLoc()[0] + x, robot.getLoc()[1] + y] != 500):
                        continue
                    else:  # assume take action

                        if (x == 0 and direction == 0):
                            currentInfo = char
                        else:
                            temp = 1
                            game.tick(robot.getLoc(), [x, direction], 1)  # assuming temporary movement
                            tempmask[robot.getLoc()[0] + x, robot.getLoc()[1] + direction] = 1

                            ### this pixel can correspond to 0 or 255 or any value in between. Lets check for 0 and 255
                            check = [0, 255]
                            diff = []
                            for k in range(len(check)):
                                tempexploredMap[robot.getLoc()[0] + x, robot.getLoc()[1] + direction] = check[k]
                                tempimage = uNet.runNetwork(tempexploredMap, tempmask)
                                # Normalize the image
                                # for row in range(28):
                                #    for col in range(28):
                                #        if (image[row, col] <= 0):
                                #            image[row, col] = 0

                                char = classNet.runNetwork(tempimage)
                                char = np.reshape(char, 10)
                                char = softmax(char)
                                # info.append(np.sum(abs(np.subtract(currentInfo, char))))
                                # info.append(char[map.number])
                                # info.append(np.max(np.subtract(currentInfo, char)))
                                diff.append(char)

                            info.append(np.sum(abs(np.subtract(diff[1], diff[0]))))
                            list.append([robot.getLoc()[0] + x, robot.getLoc()[1] + direction])

                while (len(info) != 0):
                    index = np.argmax(info)
                    pos = list[index]

                    if (game.exploredMap[pos[0], pos[1]] == 500):  # which means if that pos has been visited before, take the second best
                        break
                    else:
                        info.remove(info[index])
                        list.remove(list[index])

                game.tick(currentLoc, [pos[0]-currentLoc[0], pos[1]-currentLoc[1]], 0)  # move along that direction
                plt.plot(robot.xLoc, robot.yLoc, marker="s", color="m", linewidth=2,
                        markersize=10)  # current location of robot
                mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
                reward = reward - 1
                currentLoc = robot.getLoc()
                # direction_y = mov_y / abs(mov_y)

                image = uNet.runNetwork(game.exploredMap, mask)

                char = classNet.runNetwork(
                    image)  # what digit world belongs to: gives value corresponding to each class
                char = np.reshape(char, 10)

                char = softmax(char)

                print("Current Position of robot", [robot.xLoc, robot.yLoc], "predicted character:", char.argmax(),
                      "with prob:", char.max(), "Pixel Value:", image[robot.xLoc, robot.yLoc])


            reached = True

    if (reached == True):
        reward = reward + 100
        print("Reached Correct Goal")

    print("Reached at:", robot.getLoc(), "Steps:", steps, "Total Reward:", reward)
    print("#######################################################################################")

    #print(game.exploredMap)
    #print(mask)

    rewards.append(reward)
    Image.fromarray(image*mask).show() # show the predicted image now
    plt.show()
    plt.pause(0.0001)
    #plt.close()


########## Average out reward ##################
averageReward = np.sum(rewards)/k+1

print("Average Reward:", averageReward)