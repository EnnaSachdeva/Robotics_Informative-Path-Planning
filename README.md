# Robotics_Informative-Path-Planning

The map estimation consists of 2 networks:

• WorldEstimatingNetwork: It predicts how the world
looks like. It gives an estimate of the pixel values at
each grid cell of the entire map, based on the grid cells
which have already been traversed by the cells.

• DigitClassifcationNetwork: It provides an estimate of
what digit the world belongs. This gives the values corresponding to each class (digit from 0 to 9), where
the world might belong to. 


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
