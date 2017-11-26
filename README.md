This is a small project I started a while ago to give me something to do in my freetime.

The program computes the jacobian, that you might remember from calculus, of a computation graph containing a convolutional layer and a fully connected layer.

I just kind of threw this together after reading about conv nets at neuralnetworksanddeeplearning.com

I plan on testing the jacobian computations some more and then breaking the program up into 'layer' components (conv layer, max pool layer, fully connected layer).

I may or may not get around to doing that (of course). I'm happy that I've (seemingly) implemented the jacobians correctly. But it also would be nice to take the program a few steps further by increasing it's modularity, testability and implementing the gradient computations a bit more intelligently.
Currently, the code is in severe need of organization, testing, and documentation. Maybe I'll get around to it over xmas break.