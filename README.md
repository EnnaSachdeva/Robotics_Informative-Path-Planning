# Robotics_Informative-Path-Planning

The map estimation consists of 2 networks:
• WorldEstimatingNetwork: It predicts how the world
looks like. It gives an estimate of the pixel values at
each grid cell of the entire map, based on the grid cells
which have already been traversed by the cells.
• DigitClassifcationNetwork: It provides an estimate of
what digit the world belongs. This gives the values corresponding to each class (digit from 0 to 9), where
the world might belong to. 
