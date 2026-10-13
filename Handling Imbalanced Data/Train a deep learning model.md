### Train a deep learning model



Now prepare the data for machine learning. Under the existing code cell, use the + icon to add a new code cell. Then in the new cell, enter and run the following code to:



* Remove any incomplete rows
* Encode the (string) island name as an integer
* Apply appropriate data types
* Normalize the numeric data to a similar scale
* Split the data into two datasets: one for training, and another for testing.



from pyspark.sql.types import \*

from pyspark.sql.functions import \*

from sklearn.model\_selection import train\_test\_split



\# Load the data, removing any incomplete rows

df = spark.read.format("csv").option("header", "true").load("/deepml\_lab/penguins.csv").dropna()



\# Encode the Island with a simple integer index

\# Scale FlipperLength and BodyMass so they're on a similar scale to the bill measurements

islands = df.select(collect\_set("Island").alias('Islands')).first()\['Islands']

island\_indexes = \[(islands\[i], i) for i in range(0, len(islands))]

df\_indexes = spark.createDataFrame(island\_indexes).toDF('Island', 'IslandIdx')

data = df.join(df\_indexes, \['Island'], 'left').select(col("IslandIdx"),

&#x20;                 col("CulmenLength").astype("float"),

&#x20;                 col("CulmenDepth").astype("float"),

&#x20;                 (col("FlipperLength").astype("float")/10).alias("FlipperScaled"),

&#x20;                  (col("BodyMass").astype("float")/100).alias("MassScaled"),

&#x20;                 col("Species").astype("int")

&#x20;                  )



\# Oversample the dataframe to triple its size

\# (Deep learning techniques like LOTS of data)

for i in range(1,3):

&#x20;  data = data.union(data)



\# Split the data into training and testing datasets

features = \['IslandIdx','CulmenLength','CulmenDepth','FlipperScaled','MassScaled']

label = 'Species'



\# Split data 70%-30% into training set and test set

x\_train, x\_test, y\_train, y\_test = train\_test\_split(data.toPandas()\[features].values,

&#x20;                                                  data.toPandas()\[label].values,

&#x20;                                                  test\_size=0.30,

&#x20;                                                  random\_state=0)



print ('Training Set: %d rows, Test Set: %d rows \\n' % (len(x\_train), len(x\_test)))



###### Install and import the PyTorch libraries



PyTorch is a framework for creating machine learning models, including deep neural networks (DNNs). Since we plan to use PyTorch to create our penguin classifier, we'll need to import the PyTorch libraries we intend to use. PyTorch is already installed on Azure databricks clusters with an ML Databricks runtime (the specific installation of of PyTorch depends on whether the cluster has graphics processing units (GPUs) that can be used for high-performance processing via cuda).



import torch

import torch.nn as nn

import torch.utils.data as td

import torch.nn.functional as F



\# Set random seed for reproducability

torch.manual\_seed(0)



print("Libraries imported - ready to use PyTorch", torch.\_\_version\_\_)



###### Create data loaders

PyTorch makes use of data loaders to load training and validation data in batches. We've already loaded the data into numpy arrays, but we need to wrap those in PyTorch datasets (in which the data is converted to PyTorch tensor objects) and create loaders to read batches from those datasets.



\# Create a dataset and loader for the training data and labels

train\_x = torch.Tensor(x\_train).float()

train\_y = torch.Tensor(y\_train).long()

train\_ds = td.TensorDataset(train\_x,train\_y)

train\_loader = td.DataLoader(train\_ds, batch\_size=20,

&#x20;  shuffle=False, num\_workers=1)



\# Create a dataset and loader for the test data and labels

test\_x = torch.Tensor(x\_test).float()

test\_y = torch.Tensor(y\_test).long()

test\_ds = td.TensorDataset(test\_x,test\_y)

test\_loader = td.DataLoader(test\_ds, batch\_size=20,

&#x20;                           shuffle=False, num\_workers=1)

print('Ready to load data')



###### Define a neural network

Now we're ready to define our neural network. In this case, we'll create a network that consists of 3 fully-connected layers:



* An input layer that receives an input value for each feature (in this case, the island index and four penguin measurements) and generated 10 outputs.
* A hidden layer that receives ten inputs from the input layer and sends ten outputs to the next layer.
* An output layer that generates a vector of probabilities for each of the three possible penguin species.

As we train the network by passing data through it, the forward function will apply RELU activation functions to the first two layers (to constrain the results to positive numbers) and return a final output layer that uses a log\_softmax function to return a value that represents a probability score for each of the three possible classes.



\# Number of hidden layer nodes

hl = 10



\# Define the neural network

class PenguinNet(nn.Module):

&#x20;  def \_\_init\_\_(self):

&#x20;      super(PenguinNet, self).\_\_init\_\_()

&#x20;      self.fc1 = nn.Linear(len(features), hl)

&#x20;      self.fc2 = nn.Linear(hl, hl)

&#x20;      self.fc3 = nn.Linear(hl, 3)



&#x20;  def forward(self, x):

&#x20;      fc1\_output = torch.relu(self.fc1(x))

&#x20;      fc2\_output = torch.relu(self.fc2(fc1\_output))

&#x20;      y = F.log\_softmax(self.fc3(fc2\_output).float(), dim=1)

&#x20;      return y



\# Create a model instance from the network

model = PenguinNet()

print(model)



###### Create functions to train and test a neural network model

To train the model, we need to repeatedly feed the training values forward through the network, use a loss function to calculate the loss, use an optimizer to backpropagate the weight and bias value adjustments, and validate the model using the test data we withheld.

To do this, use the following code to create a function to train and optimize the model, and function to test the model.



def train(model, data\_loader, optimizer):

&#x20;  device = torch.device('cuda' if torch.cuda.is\_available() else 'cpu')

&#x20;  model.to(device)

&#x20;  # Set the model to training mode

&#x20;  model.train()

&#x20;  train\_loss = 0



&#x20;  for batch, tensor in enumerate(data\_loader):

&#x20;      data, target = tensor

&#x20;      #feedforward

&#x20;      optimizer.zero\_grad()

&#x20;      out = model(data)

&#x20;      loss = loss\_criteria(out, target)

&#x20;      train\_loss += loss.item()

&#x20;      # backpropagate adjustments to the weights

&#x20;      loss.backward()

&#x20;      optimizer.step()



&#x20;  #Return average loss

&#x20;  avg\_loss = train\_loss / (batch+1)

&#x20;  print('Training set: Average loss: {:.6f}'.format(avg\_loss))

&#x20;  return avg\_loss





def test(model, data\_loader):

&#x20;  device = torch.device('cuda' if torch.cuda.is\_available() else 'cpu')

&#x20;  model.to(device)

&#x20;  # Switch the model to evaluation mode (so we don't backpropagate)

&#x20;  model.eval()

&#x20;  test\_loss = 0

&#x20;  correct = 0



&#x20;  with torch.no\_grad():

&#x20;      batch\_count = 0

&#x20;      for batch, tensor in enumerate(data\_loader):

&#x20;          batch\_count += 1

&#x20;          data, target = tensor

&#x20;          # Get the predictions

&#x20;          out = model(data)

&#x20;          # calculate the loss

&#x20;          test\_loss += loss\_criteria(out, target).item()

&#x20;          # Calculate the accuracy

&#x20;          \_, predicted = torch.max(out.data, 1)

&#x20;          correct += torch.sum(target==predicted).item()



&#x20;  # Calculate the average loss and total accuracy for this epoch

&#x20;  avg\_loss = test\_loss/batch\_count

&#x20;  print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\\n'.format(

&#x20;      avg\_loss, correct, len(data\_loader.dataset),

&#x20;      100. \* correct / len(data\_loader.dataset)))



&#x20;  # return average loss for the epoch

&#x20;  return avg\_loss



###### Train a model

Now you can use the train and test functions to train a neural network model. You train neural networks iteratively over multiple epochs, logging the loss and accuracy statistics for each epoch.



\# Specify the loss criteria (we'll use CrossEntropyLoss for multi-class classification)

loss\_criteria = nn.CrossEntropyLoss()



\# Use an optimizer to adjust weights and reduce loss

learning\_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning\_rate)

optimizer.zero\_grad()



\# We'll track metrics for each epoch in these arrays

epoch\_nums = \[]

training\_loss = \[]

validation\_loss = \[]



\# Train over 100 epochs

epochs = 100

for epoch in range(1, epochs + 1):



&#x20;  # print the epoch number

&#x20;  print('Epoch: {}'.format(epoch))



&#x20;  # Feed training data into the model

&#x20;  train\_loss = train(model, train\_loader, optimizer)



&#x20;  # Feed the test data into the model to check its performance

&#x20;  test\_loss = test(model, test\_loader)



&#x20;  # Log the metrics for this epoch

&#x20;  epoch\_nums.append(epoch)

&#x20;  training\_loss.append(train\_loss)

&#x20;  validation\_loss.append(test\_loss)



While the training process is running, let's try to understand what's happening:



In each epoch, the full set of training data is passed forward through the network. There are five features for each observation, and five corresponding nodes in the input layer - so the features for each observation are passed as a vector of five values to that layer. However, for efficiency, the feature vectors are grouped into batches; so actually a matrix of multiple feature vectors is fed in each time.

The matrix of feature values is processed by a function that performs a weighted sum using initialized weights and bias values. The result of this function is then processed by the activation function for the input layer to constrain the values passed to the nodes in the next layer.

The weighted sum and activation functions are repeated in each layer. Note that the functions operate on vectors and matrices rather than individual scalar values. In other words, the forward pass is essentially a series of nested linear algebra functions. This is the reason data scientists prefer to use computers with graphical processing units (GPUs), since these are optimized for matrix and vector calculations.

In the final layer of the network, the output vectors contain a calculated value for each possible class (in this case, classes 0, 1, and 2). This vector is processed by a loss function that determines how far they are from the expected values based on the actual classes - so for example, suppose the output for a Gentoo penguin (class 1) observation is \[0.3, 0.4, 0.3]. The correct prediction would be \[0.0, 1.0, 0.0], so the variance between the predicted and actual values (how far away each predicted value is from what it should be) is \[0.3, 0.6, 0.3]. This variance is aggregated for each batch and maintained as a running aggregate to calculate the overall level of error (loss) incurred by the training data for the epoch.

At the end of each epoch, the validation data is passed through the network, and its loss and accuracy (proportion of correct predictions based on the highest probability value in the output vector) are also calculated. It's useful to do this because it enables us to compare the performance of the model after each epoch using data on which it was not trained, helping us determine if it will generalize well for new data or if it's overfitted to the training data.

After all the data has been passed forward through the network, the output of the loss function for the training data (but not the validation data) is passed to the optimizer. The precise details of how the optimizer processes the loss vary depending on the specific optimization algorithm being used; but fundamentally you can think of the entire network, from the input layer to the loss function as being one big nested (composite) function. The optimizer applies some differential calculus to calculate partial derivatives for the function with respect to each weight and bias value that was used in the network. It's possible to do this efficiently for a nested function due to something called the chain rule, which enables you to determine the derivative of a composite function from the derivatives of its inner function and outer functions. You don't really need to worry about the details of the math here (the optimizer does it for you), but the end result is that the partial derivatives tell us about the slope (or gradient) of the loss function with respect to each weight and bias value - in other words, we can determine whether to increase or decrease the weight and bias values in order to minimize the loss.

Having determined in which direction to adjust the weights and biases, the optimizer uses the learning rate to determine by how much to adjust them; and then works backwards through the network in a process called backpropagation to assign new values to the weights and biases in each layer.

Now the next epoch repeats the whole training, validation, and backpropagation process starting with the revised weights and biases from the previous epoch - which hopefully will result in a lower level of loss.

The process continues like this for 100 epochs.



###### Review training and validation loss

After training is complete, we can examine the loss metrics we recorded while training and validating the model. We're really looking for two things:



The loss should reduce with each epoch, showing that the model is learning the right weights and biases to predict the correct labels.

The training loss and validation loss should follow a similar trend, showing that the model is not overfitting to the training data.



Use the following code to plot the loss:



%matplotlib inline

from matplotlib import pyplot as plt



plt.plot(epoch\_nums, training\_loss)

plt.plot(epoch\_nums, validation\_loss)

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(\['training', 'validation'], loc='upper right')

plt.show()



###### View the learned weights and biases



The trained model consists of the final weights and biases that were determined by the optimizer during training. Based on our network model we should expect the following values for each layer:



Layer 1 (fc1): There are five input values going to ten output nodes, so there should be 10 x 5 weights and 10 bias values.

Layer 2 (fc2): There are ten input values going to ten output nodes, so there should be 10 x 10 weights and 10 bias values.

Layer 3 (fc3): There are ten input values going to three output nodes, so there should be 3 x 10 weights and 3 bias values.



Use the following code to view the layers in your trained model:

for param\_tensor in model.state\_dict():

&#x20;  print(param\_tensor, "\\n", model.state\_dict()\[param\_tensor].numpy())



###### Save and use the trained model



Now that we have a trained model, we can save its trained weights for use later.



Use the following code to save the model:

\# Save the model weights

model\_file = '/dbfs/penguin\_classifier.pt'

torch.save(model.state\_dict(), model\_file)

del model

print('model saved as', model\_file)



Use the following code to load the model weights and predict the species for a new penguin observation:

\# New penguin features

x\_new = \[\[1, 50.4,15.3,20,50]]

print ('New sample: {}'.format(x\_new))



\# Create a new model class and load weights

model = PenguinNet()

model.load\_state\_dict(torch.load(model\_file))



\# Set model to evaluation mode

model.eval()



\# Get a prediction for the new data sample

x = torch.Tensor(x\_new).float()

\_, predicted = torch.max(model(x).data, 1)



print('Prediction:',predicted.item())

