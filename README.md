# Robotics_Informative-Path-Planning

run main_greedy.py #for greedy informative path planning
run main_sophisticated.py # for sophisticated non-greedy path plannign

This code was written in python 3.6.5
To execute this code, you will need Python 3.5+ (I recommend downloading Anaconda) and pytorch
pytorch can be downloaded at https://pytorch.org

pytorch install instructions:
1. go to https://pytorch.org
2. Click the Òget startedÓ button
3. selection the appropriate settings (I would recommend: Stable, OS, Conda, python 3.6, None)
4. Run the Òpython main.pyÓ to test

### Structure of code

Starting at (0,0), navigate to a specific corner of the map. The goal depends on what world you occupy. The worlds are constructed from the MNIST dataset, a hand drawn number dataset consisting of numbers 0-9. the grid world is 28x28
if the MNIST digit is: 

digit: 0-2 - goal: (0,27)
digit: 3-5 - goal: (27,27)
digit: 6-9 - goal: (27,0)

The robot can travel one space at a time and can observe the value of the spaces it has visited. It costs -1 reward to move and -400 reward if you travel to the wrong goal. Reaching the correct goal provides +100 reward, at which point the mission ends. You will design algorithms to minimize your cost to reach the correct goal.  We have provided two trained neural networks to help. One network provides an estimate of what the world looks like given what you have currently observed. The other network takes this estimate and provides an estimate of what digit the world belongs to.


FunctionList provides 3 classes:

Class Map:
	Class Variables:
		map - saves a numpy array of an MNIST digit
		number - the number represented by the MNIST image

	Functions:
		GetNewMap - sets map to the next numpy array from the MNIST dataset


Class WorldEsimtateNetwork:
	Class Variables:
		network - This is the network that will provide an estimate of what the world looks like

	Functions:
		runNetwork - requires two inputs. The first is the map at its current level of exploration. Second is a binary mask the same size of the map, with 1's for area's explored and 0's for unexplored regions

		This function returns a numpy array of the current estimate of what the world looks like

Class DigitClassifcationNetwork:
	Class Variables:
		network - This is the network that will provide an estimate of what Digit the world is

	Functions:
		runNetwork - Single input, the estimated world provided by WorldEstimateNetwork

This function returns an array of size 10, with each index representing the different digits the world could be. The largest number represents the networks estimate of what class the world is.

	
The map estimation consists of 2 networks:

• WorldEstimatingNetwork: It predicts how the world
looks like. It gives an estimate of the pixel values at
each grid cell of the entire map, based on the grid cells
which have already been traversed by the cells.

• DigitClassifcationNetwork: It provides an estimate of
what digit the world belongs. This gives the values corresponding to each class (digit from 0 to 9), where
the world might belong to. 

### Greedy Algorithm for Path planning:
For this case, the output of the WorldEstimatingNetwork
has been normalized to values lesser than 0 has been normalized to 0. The information quality from neural network
is obtained from the ouput of the DigitClassifcationNetwork,
which gives a prediction of class from 0 to 9, to which the
world belongs. Taking softmax of these 10 values provide
the probabilities of each of these class Since the robot should look one step ahead and move
greedily based on the maximal information of the prediction
of Neural Network. This means if robot is at (0, 0), it
can look ahead at 2 positions, i.e Right and Down. Since
the WorldEstimatingNetwork predicts the world with the
knowledge of the grid cells that have already been traversed
and the corresponding original pixel values in those cells.
However, robot does not have information of the original
pixel values in its immediate neighbors, until it has actually
traversed there. Therefore, robot makes a prediction of 2
extreme values of the pixel values in its neighboring cells,
i.e 0 and 255. For each neighboring cell being 0 and 255,
the Neural Network gives the softmax prediction of classes
of the image. The difference between the softmax value
corresponding to the neighboring pixel being 0 or 255,
gives the quality of information in the neighboring pixel.If
the difference is too less or close to 0, then the neighboring
pixel value being 0 or 255, does not matter in the overall
prediction of the image and hence, it contains no useful
information which could help in the overall prediction of the
digit which the world belongs to. However, if the difference
is more, this contains more information which could help
with the prediction of the image. This naturally makes the
robot move in the direction of maximal information.

### Sophisticated Algorithm for Path planning:

For this algorithm too, all negative values of estimated
output from WorldEstimatingNetwork has been normalized
to 0 and this resulting predicted image is further fed to the
DigitClassifcationNetwork. In contrast to the greedy algorithm explained before, this approach will just use the prediction output of WorldEstimatingNetwork to determine the
direction of movement. The algorithm works as following-

• This algorithm is based on following the nearest maximum white/gray pixel value. white pixels in the predicted output.

• It will search the nearest predicted pixel values greater
than or equal to some threshhold (threshhold = 100).

• As soon as it predicts this pixel, it finds a path towards
that pixel with maximum number of non-zero pixels
values. There can be cases, when the robot can get stuck
in the local minima, where it will just move around
those same high values pixels grids.

• To eliminate such cases, a buffer (matrix) keeps a count
of the grid cells traversed before and promotes moving
towards those grid cells, which have not been traversed
before.

• In cases when all grid cells nearest to the pixels are
already traversed, however the prediction is still lesser
than 75%, the robot extends its search space and finds
the white pixels in that search space.

• However, there can be cases where robot might still not
be able to find the white pixels, not traversed before,
then the robot searches for pixels values which are
lesser than threshhold values. Though, it will make the
movement towards slightly darker pixels, however it
will make the prediction more accurate. This adaptive
threshhold acts as heuristics in this algorithm.

• Further to eliminate false positive cases, where the
neural network accidently predicts the right digit once,
I have kept a count which ensures that if the softmax
output is greater than 75% for at least 10 consequent
cases, the iterations should keep on continuing.
