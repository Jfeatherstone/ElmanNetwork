import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(output, target):
    return np.sum(0.5 * (target - output)**2)

def normalize(data, normMin=.15, normMax=.85):
    normalizedData = data - np.min(data)
    normalizedData = normalizedData / np.max(normalizedData) * (normMax - normMin) + normMin
    
    return normalizedData

class ElmanNetwork():
    
    def __init__(self, inputDim=None, contextDim=None, outputDim=None, learningRate=None, loadPath=None):
        
        if not loadPath:
            self.inputDim = inputDim
            self.contextDim = contextDim # Aka hidden layer dim
            self.outputDim = outputDim
            self.learningRate = learningRate
                        
            self.initialize()
        else:
            self.load(loadPath)
            
    def initialize(self):
        """
        Initialize the weights and biases by sampling from a 
        standard normal distribution.
        """
        self.Wux = np.random.normal(size=(self.contextDim, self.inputDim))
        self.Wuc = np.random.normal(size=(self.contextDim, self.contextDim))
        self.Wvc = np.random.normal(size=(self.outputDim, self.contextDim))
        
        self.bu = np.random.normal(size=(self.contextDim))
        self.bv = np.random.normal(size=(self.outputDim))
        
        self.activation = sigmoid
        self.error = mse
            
        # Set all gradients to zero
        self._resetGradients()
    
    def _forwardStep(self, inputArr, prevContextArr):
        """
        Perform a single forward pass of the network, calculating the output and
        values of context neurons.
        
        Parameters
        ----------
        inputArr : np.ndarray[self.inputDim,]
            The input values for the current time step, t.
            
        prevContextArr : Np.ndarray[self.contextDim,]
            The context neuron values for the previous time step, t - 1.
            
        Returns
        -------
        c : np.ndarray[self.contextDim,]
            The context neuron values for the current time step, t.
            
        o : np.ndarray[self.outputDim,]
            The output values for the current time step, t.
        """
        u = self.Wux @ inputArr + self.Wuc @ prevContextArr + self.bu
        c = self.activation(u)
        v = self.Wvc @ c + self.bv
        o = self.activation(v)
        return c, o
    
    
    def forwardSequence(self, inputArr):
        """
        Compute the output and context values for an input sequence.
        
        Parameters
        ----------
        inputArr : np.ndarray[T, self.inputDim]
            The input sequence for T timesteps
            
        Returns
        -------
        contextArr : np.ndarray[T, self.contextDim]
            The values of the context neurons at each timestep.
            
        outputArr : np.ndarray[T, self.outputDim]
            The output values at each timestep.
        """

        contextArr = np.zeros((len(inputArr), self.contextDim))
        outputArr = np.zeros((len(inputArr), self.outputDim))
        
        for i in range(1, len(inputArr)):
            # Reshape part adds [[ ]] around the value (or as many brackets are needed
            # to end up with two around the elements)
            c, o = self._forwardStep(inputArr[i].reshape(self.inputDim), contextArr[i-1])
            contextArr[i] = c
            outputArr[i] = o
                    
        return contextArr, outputArr
        
        
    def _resetGradients(self):
        """
        Set all gradients to zero, presumably after updating learning parameters.
        """
        # Set all of the gradients to zero
        self.dEdWuc = np.zeros_like(self.Wuc)
        self.dEdWux = np.zeros_like(self.Wux)
        self.dEdWvc = np.zeros_like(self.Wvc)
        self.dEdbu = np.zeros_like(self.bu)
        self.dEdbv = np.zeros_like(self.bv)
    
    
    def _backwardStep(self, inputArr, contextArr, outputArr, targetArr, prevContextArr, prevdEdu):
        """
        Compute a single backward step for the backpropagation.
        
        Parameters
        ----------
        inputArr : np.ndarray[self.inputDim,]
            The input to the network at the current time step, t.
        
        contextArr : np.ndarray[self.contextDim,]
            The context layer values for the network at the current time
            step, t.

        outputArr : np.ndarray[self.outputDim,]
            The output of the network at the current time step, t.
            
        targetArr : np.ndarray[self.outputDim,]
            The target output for the network at the current time step, t.
                        
        prevContextArr : np.ndarray[self.contextDim,]
            The context layer values for the network at the 'previous' current
            time step, t + 1 (since we are going backwards).

        prevdEdu : np.ndarray[self.contextDim,]
            The gradient of the error with respect to the inputs for the
            'previous' time step, t + 1 (since we are going backwards). If
            on the final timestep, `np.zeros((self.contextDim, 1))`
            should be passed.
        """
        #print(inputArr.shape)
        #print(contextArr.shape)
        #print(outputArr.shape)
        #print(targetArr.shape)
        #print(prevContextArr.shape)
        #print(prevdEdu.shape)
        
        # Derivative of mse is just difference
        dEdy = outputArr - targetArr
                
        # Comes from derivative of the sigmoid
        dEdv = dEdy * (1 - outputArr) * outputArr
        
        # Has two contributions: one from current step and one from previous
        # Note that 'prevdEdu' is actually the derivative for the
        # next step in time, since we are going backwards
        dEdc = self.Wvc.T @ dEdv + self.Wuc.T @ prevdEdu
        
        # Now calculate this derivative for the current timestep
        dEdu = dEdc * (1 - contextArr) * contextArr
        
        #print(dEdv.shape)
        #print(contextArr.T.shape)
        # Now we can calculate the gradients for our learnable
        # parameters from the ones above
        # We want to keep track of these as cumulative values
        # Weights
        self.dEdWvc += dEdv @ contextArr.T
        self.dEdWuc += dEdu @ prevContextArr.T
        self.dEdWux += dEdu @ inputArr.T
        # Biases
        #print(dEdv.shape)
        #print(self.dEdbv.shape)
        self.dEdbv += dEdv.reshape(self.dEdbv.shape)
        self.dEdbu += dEdu.reshape(self.dEdbu.shape)
        
        # We need this back at the end, otherwise we can't compute the next step
        return dEdu
    
    
    def backwardSequence(self, inputArr, contextArr, outputArr, targetArr):
        """
        Compute the backpropagation computation through the network.
        
        Parameters
        ----------
        inputArr : np.ndarray[T, self.inputDim]
            The input array at each timestep.
            
        contextArr : np.ndarray[T, self.contextDim]
            The values of the context neurons at each timestep.
            
        outputArr : np.ndarray[T, self.outputDim]
            The output of the network at each timestep.
            
        targetArr : np.ndarray[T, self.outputDim]
            The target output at each timestep.
            
        Returns
        -------
        error : np.ndarray[T]
            The MSE at each time step.
        """
        #print(inputArr.shape)
        #print(contextArr.shape)
        #print(outputArr.shape)

        # Number of timesteps
        T = inputArr.shape[0]
        
        # Add a plus 1 so that we can pass zeros to the last
        # step without extra work (and we clip it off later)
        dEdu = np.zeros((T+1, self.contextDim))
        error = 0
        
        # Reset gradients
        self._resetGradients()
        
        # Count down from last time step to first and compute gradients
        for i in range(T)[::-1]:
            # Compute gradients for learnable quantities
            dEdu[i] = self._backwardStep(inputArr[i].reshape(-1, 1),
                                         contextArr[i].reshape(-1, 1),
                                         outputArr[i].reshape(-1, 1),
                                         targetArr[i].reshape(-1, 1),
                                         contextArr[i+1].reshape(-1, 1) if i < T-1 else np.zeros((self.contextDim, 1)),
                                         dEdu[i+1].reshape(-1, 1))[:,0]
            # Compute MSE error from output and target
            error += mse(outputArr, targetArr)
            
        # Clip off the extra entry (see above)
        # Not totally necessary since we don't do anything with it, but :/
        dEdu = dEdu[:-1]
        
        return error
        
    def updateParameters(self):
        """
        Update the learning parameters from the stored gradients
        (up to first order).
        """
        # Update via (first order) gradient descent
        # Weights
        self.Wvc -= self.dEdWvc * self.learningRate
        self.Wuc -= self.dEdWuc * self.learningRate
        self.Wux -= self.dEdWux * self.learningRate
        # Biases
        self.bv -= self.dEdbv * self.learningRate
        self.bu -= self.dEdbu * self.learningRate
        
        # And reset gradients so we don't accidentally update
        # twice
        self._resetGradients()
    
        
    def predict(self, inputArr, predictionSteps=1):
        """
        Predict the next point(s) in a series using the trained
        parameters.
        
        Parameters
        ----------
        inputArr : np.ndarray[self.inputDim,]
            The single time step to predict from.
        
        predictionSteps : int
            The number of steps to predict into the future.
            
        Returns
        -------
        outputArr : np.ndarray or float
            Returns the sequence of predicted points (including the original
            input point).
        """
        outputArr = np.zeros((predictionSteps+1,self.outputDim))
        contextArr = np.zeros((predictionSteps+1,self.contextDim))
        
        outputArr[0] = inputArr
        for i in range(1, predictionSteps+1):
            contextArr[i], outputArr[i] = self._forwardStep(outputArr[i-1], contextArr[i-1])
        
        return outputArr
        
    
    def save(self, file):
        """
        Saves the trained network parameters to a file.
        
        Parameters
        ----------
        file : str
            The filename to save the model to; should be a .npz file.
        """
        np.savez(file=file, Wux=self.Wux, Wuc=self.Wuc, Wvc=self.Wvc, bu=self.bu, bv=self.bv,
                 hyperparameters=[self.inputDim, self.contextDim, self.outputDim, self.learningRate])
        
    
    def load(self, file):
        params = np.load(file)
        
        self.inputDim = int(params["hyperparameters"][0])
        self.contextDim = int(params["hyperparameters"][1])
        self.outputDim = int(params["hyperparameters"][2])
        self.learningRate = params["hyperparameters"][3]

        self.initialize()
        
        self.Wux = params["Wux"]
        self.Wuc = params["Wuc"]
        self.Wvc = params["Wvc"]
        
        self.bu = params["bu"]
        self.bv = params["bv"]