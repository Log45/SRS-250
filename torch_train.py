import torch
import torch.nn as nn
import pandas as pd
from sklearn.datasets import make_circles, make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_models import *
from helper_functions import plot_predictions, plot_decision_boundary

# Device agnostic code:
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def train_linear():
    """"""
    # Create *known* parameters
    weight = 0.7
    bias = 0.3

    # Create
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    # Create a train/test split
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split].to(device), y[:train_split].to(device)
    X_test, y_test = X[train_split:].to(device), y[train_split:].to(device)

    model = LinearRegressionModel().to(device)

    # Setup a loss function
    loss_fn = nn.L1Loss()

    # Setup an optimizer (stochastic gradient descent)
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.01) # lr = learning rate = possibly the most important hyperparameter you can set
    # An epoch is one loop through the data (this is a hyperparameter because we've set it ourselves)
    epochs = 1000

    epoch_count = []
    loss_values = []
    test_loss_values = []

    ### Training
    # 0. Loop through the data
    for epoch in range(epochs):
        # Set the model to training mode
        model.train() # train mode in PyTorh sets all parameters that require gradients to require gradients.

        # 1. Forward pass
        y_pred = model(X_train)

        # 2. Calculate the loss
        loss = loss_fn(y_pred, y_train)
        #print(f"Loss: {loss}")

        # 3. Optimizer zero grad
        optimizer.zero_grad() # start fresh each iteration of the loop

        # 4. Perform backpropagration on the loss with respect to the parameters of the model
        loss.backward()

        #5. Step the optimizer (perform gradient descent)
        optimizer.step() # y default, how the optimizer changes will accumulate through the loop; so we have to zero them above in step 3 for the loop

        ### Testing
        model.eval() # turns off things not needed for testing (dropout and batchnorm layers)
        with torch.inference_mode(): # turns off gradient tracking and more things behind the scenes
            # 1. Do the forward pass
            test_pred = model(X_test)

            # 2. Calculate the loss
            test_loss = loss_fn(test_pred, y_test)
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
            print(model.state_dict())
    print(X_test.shape, y_test.shape, X_train.shape, y_train.shape, y_pred.shape)
    plot_predictions(X_train.to("cpu").detach(), y_train.to("cpu").detach(), X_test.to("cpu").detach(), y_test.to("cpu").detach(), test_pred.to("cpu").detach())



def train_binary():
    # Make 1000 samples

    n_samples = 1000

    # Create circles

    X, y = make_circles(n_samples,
                        noise = 0.03,
                        random_state = 42)

    circles = pd.DataFrame({"X1": X[:, 0],
                            "X2": X[:, 1],
                            "label": y})

    # Visualize

    plt.scatter(x=X[:, 0],
                y = X[:, 1],
                c = y,
                cmap=plt.cm.RdYlBu)

    plt.show()

    # Turn data into tensors
    X = torch.from_numpy(X).type(torch.float) # turn it into default type to avoid errors
    y = torch.from_numpy(y).type(torch.float)

    # Split data into training and test sets

    # Train data, test data, train labels, test labels
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.2, # 20% of data will be tested
                                                        random_state = 42)

    model_0 = ClassificationModel(2, 1, 32).to(device)
    #model_0 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1)).to(device)

    # Setup the loss function
    # nn.BCELoss() is another option: requires inputs to have gone through the sigmoid activation function prior to input to BCELoss
    #loss_fn = nn.BCELoss() 
    loss_fn=nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(params=model_0.parameters(),
                                lr = 0.1)

    # Calculate accuracy - out of 100 examples, what percentage does our model get right?
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct/len(y_pred)) * 100
        return acc

    # Set the number of epochs
    epochs = 1000

    # Put data to target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Build training and evaluation loop
    for epoch in range(epochs):
        ### Training
        model_0.train()

        # 1. Forward pass
        y_logits = model_0(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits into pred probs into pred labels

        # 2. Calculate loss/accuracy
        # loss = loss_fn(torch.sigmoid(y_logits), y_train): nn.BCELoss expects prediction probabilities as input
        loss = loss_fn(y_logits, # nn.BCEWithLogitsLoss expects raw logits as input
                        y_train)
        acc = accuracy_fn(y_true=y_train,
                            y_pred=y_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward (backpropagation)
        loss.backward()

        # 5. Optimizer step (gradient descent)
        optimizer.step()

        ### Testing
        model_0.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model_0(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))

            # 2. Calculate test loss/acc
            test_loss = loss_fn(test_logits, test_pred)
            test_acc = accuracy_fn(y_test, test_pred)

        # Print out what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

    # Plot decision boundary of the model
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_0, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_0, X_test, y_test)
   #plot_predictions()
    
def train_multiclass():
    # Set the hyperparameters for data creation
    NUM_CLASSES = 4
    NUM_FEATURES = 2
    RANDOM_SEED = 42

    # 1. Create multi-class data
    X_blob, y_blob = make_blobs(n_samples=1000,
                                n_features=NUM_FEATURES,
                                centers=NUM_CLASSES,
                                cluster_std=1.5, # give clusters a little shake up
                                random_state=RANDOM_SEED
                                )

    # 2. Turn data into tensors
    X_blob = torch.from_numpy(X_blob).type(torch.float)
    y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

    # 3. Split into train and test
    X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                            y_blob,
                                                                            test_size=0.2,
                                                                            random_state=RANDOM_SEED)

    # 4. Plot data
    plt.figure(figsize=(10, 7))
    plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)

    model = ClassificationModel(2, 4, 64).to(device)

    # Setup the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Setup the optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr = 0.1)

    # Calculate accuracy - out of 100 examples, what percentage does our model get right?
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct/len(y_pred)) * 100
        return acc
    
    # Set the number of epochs
    epochs = 1000

    # Put data to target device
    X_train, y_train = X_blob_train.to(device), y_blob_train.to(device)
    X_test, y_test = X_blob_test.to(device), y_blob_test.to(device)

    # Build training and evaluation loop
    for epoch in range(epochs):
        ### Training
        model.train()

        # 1. Forward pass
        y_logits = model(X_train).squeeze()
        y_pred_probs = torch.softmax(y_logits, dim=1) # turn logits into pred probs
        y_preds = torch.argmax(y_pred_probs, dim=1) # turn pred probs into pred labels

        # 2. Calculate loss/accuracy
        # loss = loss_fn(torch.sigmoid(y_logits), y_train): nn.BCELoss expects prediction probabilities as input
        loss = loss_fn(y_logits,
                        y_train)
        acc = accuracy_fn(y_true=y_train,
                            y_pred=y_preds)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward (backpropagation)
        loss.backward()

        # 5. Optimizer step (gradient descent)
        optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test).squeeze()
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

            # 2. Calculate test loss/acc
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_test, test_pred)

        # Print out what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_blob_train, y_blob_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_blob_test, y_blob_test)

if __name__ == "__main__":
    train_linear()
    train_binary()
    train_multiclass()
