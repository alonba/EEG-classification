# Classification of motor imagery EEG signals

In this project I've tried to improve the classification accuracy of an existing neural network.

The NN predicts, based only on EEG recordings, which hand the subject IMAGINED he is moving (or if he doesn't move at all).

For improving the accuracy of the NN, I used the ICA algorithm to find interesting components of the data, which contain important data regarding the classification of the movement.

Then, I recreated the electrodes data using only these components. This way, we feed the neural network with data that is much more relevant.

Eventually, I was able to improve the classification accuracy from 70% to 90%. 
