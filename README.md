## Elman RNN

Simple implementation of a recurrent neural network with one context layer, with backpropagation calculation performed explicitly.

### Usage

There are two main implementations of the Elman RNN: [`elman.py`](https://github.com/Jfeatherstone/ElmanNetwork/blob/master/elman.py) and [`elman_opt.py`](https://github.com/Jfeatherstone/ElmanNetwork/blob/master/elman.py). The former is a basic implementation of the network, and the latter uses [numba](https://numba.pydata.org/) to compile the backpropagation calculation in C, giving about a 5x reduction in calculation time. Practically, both are completely interchangeable, and the relevant class and functions can be imported as:

```
from <elman | elman_opt> import ElmanNetwork, normalize, load, save
```

For more info, see the example notebooks, listed below.

### Examples

#### [Drawing a circle](https://github.com/Jfeatherstone/ElmanNetwork/blob/master/MouseCircle.ipynb)

![circle results](https://github.com/Jfeatherstone/ElmanNetwork/blob/master/images/circle_cl.png)

#### [Drawing a figure-eight](https://github.com/Jfeatherstone/ElmanNetwork/blob/master/MouseFigureEight.ipynb)

![figure-eight results](https://github.com/Jfeatherstone/ElmanNetwork/blob/master/images/figure_eight_cl.png)

#### [Drawing both a circle and figure-eight (multi-attractor)](https://github.com/Jfeatherstone/ElmanNetwork/blob/master/MultiAttractor.ipynb)

![multi-attractor learning curve](https://github.com/Jfeatherstone/ElmanNetwork/blob/master/images/multi_attractor_learning_curve.png)

![multi-attactor phase diagram](https://github.com/Jfeatherstone/ElmanNetwork/blob/master/images/multi_attractor_phase_diagram.png)

### References

Elman, Jeffrey L. “Distributed Representations, Simple Recurrent Networks, and Grammatical Structure.” Machine Learning 7, no. 2 (September 1, 1991): 195–225. https://doi.org/10.1007/BF00114844.

