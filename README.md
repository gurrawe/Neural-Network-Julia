# Neural-Network-Julia
An implementation of a neural network in Julia.

The parameter file 
To start the program a parameter file needs to be inputted. There is an example parametersfile 
submitted in the repo. The structure of the file looks like this: 
- The name of the network. Example: NeuralNet. 
- The amount of test data compared to total data. Example: 0.15. 
- The name of the data-file. Example: turbine.txt. 
- Batch size. Example: 1. 
- Number of epochs. Example: 500. 
- Learning rate. Example 0.2. 
- Momentum. Example 0.5. 
- Numbers of layers in the network. Example 4 
- The following lines are the sizes of each layer starting from layer 1 
The program can not handle if the parameter file is wrongly inputted. Either if the number of 
neurons in the first layer is set to a different number then the input data.


 
