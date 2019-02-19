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


rewards = []
times = []
for k in range(0,10):
    map = Map(k)
    #map = MapgetNewMap()
    # Maximum information corresponds to 255 and min to 0
    data = map.map  # 28*28 map

    print(map.number) # number represented by the map

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

    reward = 0
    mask = np.zeros((28, 28))
    tempmask = np.zeros((28, 28))

    # intialize movement to update the mask and exploredMap
    game.tick(robot.getLoc(), [0,0], 0)
    mask[robot.getLoc()[0], robot.getLoc()[1]] = 1


    steps = 0
    count = 0

    t = time.time()
    for i in range(1000): # movements till it reaches the goal
        time.sleep(0.1)
        currentPos = [robot.getLoc()[0], robot.getLoc()[1]]

        image = uNet.runNetwork(game.exploredMap, mask)
        #Image.fromarray(image).show()
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
        print("Current Position of robot", [robot.xLoc, robot.yLoc], "predicted character: ", char.argmax(), char.max())


        if (char[map.number] > 0.75): # if sufficient exploration happened
            count = count + 1

        if count > 5:
            explored = True
            print("explored")
            break

        plt.plot(robot.xLoc, robot.yLoc,  marker="s", color="c", linewidth=2, markersize=10) # current location of robot
        draw()
        #plt.ion()
        #plt.show()

        # robot looks one step ahead
        list = []
        info = []
        pixels = []
        temp = 0

        # move greedily towards white pixels
        found = 0

        currentInfo = image[robot.getLoc()[0], robot.getLoc()[1]]

        lookahead = 1

        while (len(list) == 0):

            for x in range(-lookahead, lookahead+1):
                for y in range(-lookahead, lookahead+1):
                    if (robot.getLoc()[0] + x < 0) or (robot.getLoc()[0] + x > 27) or (robot.getLoc()[1] + y < 0) \
                                or (robot.getLoc()[1] + y > 27) or (mask[robot.getLoc()[0] + x, robot.getLoc()[1] + y] == 1):
                        continue



                    #if(robot.getLoc()[0] == 27) or (robot.getLoc()[1] == 27):
                    #    notfound = True
                     #   break
                    else:  # take action
                        if (image[robot.getLoc()[0] + x, robot.getLoc()[1] + y] > 100):
                            pixels.append(image[robot.getLoc()[0] + x, robot.getLoc()[1] + y])
                            list.append([robot.getLoc()[0] + x, robot.getLoc()[1] + y])

            lookahead = lookahead + 1   # search for the next neighbors
            notfound = False
            if (notfound == True):
                break

        index = np.argmax(pixels)
        pos = list[index]


        # perform a temporary goal directed operation

        currentLoc = robot.getLoc()
        while (currentLoc[0] != pos[0]):
            time.sleep(0.01)
            steps = steps + 1
            mov_x = pos[0] - currentLoc[0]  # cells along x axis
            direction = mov_x / abs(mov_x) # direction of movement
            game.tick(currentLoc, [direction, 0], 0) # move along that direction
            mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
            currentLoc = robot.getLoc()
            # check if it reached the correct goal
            reached = False
            if (robot.getLoc()[0] == goalLoc[0]) and (robot.getLoc()[1] == goalLoc[1]):
                reached = True
                break



            # check if it accidently moved to any of the wrong goals
            wrong = False
            for m in range(len(wrongGoals)):
                if (robot.getLoc()[0] == wrongGoals[m][0]) and (robot.getLoc()[1] == wrongGoals[m][1]):
                    wrong = True
                    break
            if (wrong == True):
                break # break the while loop


            # Reward structure
            reward = reward - 1

            mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
            plt.plot(robot.xLoc, robot.yLoc,  marker="s", color="c",linewidth=2, markersize=10)  # current location of robot
            draw()


        if(reached == True) or (wrong == True):
            break  # break the overall operation



        while(currentLoc[1] != pos[1]):
            time.sleep(0.01)
            steps = steps + 1
            mov_y = pos[1] - currentLoc[1]  # cells along x axis
            direction = mov_y / abs(mov_y)
            game.tick(currentLoc, [0, direction], 0)
            currentLoc = robot.getLoc()
            mask[robot.getLoc()[0], robot.getLoc()[1]] = 1

            reached = False
            if (robot.getLoc()[0] == goalLoc[0]) and (robot.getLoc()[1] == goalLoc[1]):
                reached = True
                break

            # check if it accidently moved to wrong goal
            wrong = False
            for m in range(len(wrongGoals)):
                if (robot.getLoc()[0] == wrongGoals[m][0]) and (robot.getLoc()[1] == wrongGoals[m][1]):
                    wrong = True
                    break
            if (wrong == True):
                break

            # Reward structure
            reward = reward - 1

            plt.plot(robot.xLoc, robot.yLoc, marker="s", color="c", linewidth=2, markersize=10)  # current location of robot




    ############## Reached either correct goal or wrong Goal #############

    print("Exploration ended at:", robot.getLoc())

    if (wrong == True): # Reached wrong Goal
        print("Reached Wrong Goal, Progressing towards actual goal now")
        reward = reward - 400
        reached = False
        # towards correct goal

        while (currentLoc[0] != goalLoc[0]):
            time.sleep(0.01)

            steps = steps + 1
            mov_x = goalLoc[0] - currentLoc[0]  # cells along x axis
            direction = mov_x / abs(mov_x)  # direction of movement

            game.tick(currentLoc, [direction, 0], 0)  # move along that direction
            mask[robot.getLoc()[0], robot.getLoc()[1]] = 1

            reward = reward - 1
            currentLoc = robot.getLoc()
            mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
            plt.plot(robot.xLoc, robot.yLoc, marker="s", color="m", linewidth=2, markersize=10)  # current location of robot
            draw()


        while (currentLoc[1] != goalLoc[1]):
            time.sleep(0.01)

            steps = steps + 1
            mov_y = goalLoc[1] - currentLoc[1]  # cells along x axis
            direction = mov_y / abs(mov_y)  # direction of movement

            game.tick(currentLoc, [direction, 0], 0)  # move along that direction
            mask[robot.getLoc()[0], robot.getLoc()[1]] = 1

            reward = reward - 1
            currentLoc = robot.getLoc()
            plt.plot(robot.xLoc, robot.yLoc, marker="s", color="m", linewidth=2, markersize=10)  # current location of robot
            draw()
        reached = True


    elif(reached == False) and (explored == True): #exploration done but landed on a different grid cell
        print('Exploration done, now progressing towards goal')

        currentLoc = robot.getLoc()
        while (currentLoc[0]!=goalLoc[0]) or (currentLoc[1]!=goalLoc[1]):
            if (currentLoc[0] != goalLoc[0]):
                steps = steps + 1
                mov_x = goalLoc[0] - currentLoc[0]  # cells along x axis
                direction = np.int(mov_x / abs(mov_x))  # direction of movement
                game.tick(currentLoc, [direction, 0], 0)  # move along that direction
                plt.plot(robot.xLoc, robot.yLoc, marker="s", color="m", linewidth=2,
                        markersize=10)  # current location of robot
                mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
                reward = reward - 1
                currentLoc = robot.getLoc()

            if (currentLoc[1] != goalLoc[1]):
                steps = steps + 1
                mov_y = goalLoc[1] - currentLoc[1]
                direction = np.int(mov_y / abs(mov_y))  # direction of movement
                game.tick(currentLoc, [0, direction], 0)  # move along that direction
                plt.plot(robot.xLoc, robot.yLoc, marker="s", color="m", linewidth=2,
                         markersize=10)  # current location of robot
                mask[robot.getLoc()[0], robot.getLoc()[1]] = 1
                reward = reward - 1
                currentLoc = robot.getLoc()
                # direction_y = mov_y / abs(mov_y)
            reached = True




    if (reached == True):
        reward = reward + 100
        print("Reached Correct Goal")

    print("Reached at:", robot.getLoc(), "Steps:", steps, "Total Reward:", reward, "Reward per time step:", reward / steps)
    print("#######################################################################################")

#print(game.exploredMap)
    #print(mask)
    elapsed = time.time() - t
    times.append(elapsed)
    rewards.append(reward)
    plt.show()
    plt.pause(0.0001)
    # plt.close()


########## Average out reward ##################
averageReward = np.sum(rewards)/k+1
averageTime = np.sum(times)/k+1
print("Average Reward:", averageReward, "Average Time:", averageTime)