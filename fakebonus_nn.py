# Imports
import torch
from torch.autograd import Variable
import numpy as np

def fakebonus_nn():
    training_set_np = np.load("training_set.npy")
    validation_set_np = np.load("validation_set.npy")
    testing_set_np = np.load("testing_set.npy")

    training_label_np = np.load("training_label.npy")
    validation_label_np = np.load("validation_label.npy")
    testing_label_np = np.load("testing_label.npy")

    torch.manual_seed(42)

    dim_x = training_set_np.shape[1]
    dim_h0 = 1024
    dim_h1 = 512
    dim_h2 = 256
    dim_out = 2

    x = Variable(torch.from_numpy(training_set_np), requires_grad=False).type(torch.FloatTensor)
    y_classes = Variable(torch.from_numpy(np.argmax(training_label_np, 1)), requires_grad=False).type(torch.LongTensor)

    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h0),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h0, dim_h1),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h1, dim_h2),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h2, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate, max_iter = 1e-5, 500
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(max_iter):
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)
        
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                           # make a step

        if t % 10 == 0 or t == max_iter - 1:
            print("Epoch: " + str(t))

            # Training Performance
            x_train = Variable(torch.from_numpy(training_set_np), requires_grad=False).type(torch.FloatTensor)
            y_pred = model(x_train).data.numpy()
            train_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(training_label_np, 1))) * 100
            print("Training Set Performance  : " + str(train_perf_i) + "%")      

            # Validation Performance  
            x_test = Variable(torch.from_numpy(validation_set_np), requires_grad=False).type(torch.FloatTensor)
            y_pred = model(x_test).data.numpy()
            valid_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(validation_label_np, 1))) * 100
            print("Validation Set Performance:  " + str(valid_perf_i) + "%\n")

    # Testing Performance
    x_test = Variable(torch.from_numpy(testing_set_np), requires_grad=False).type(torch.FloatTensor)
    y_pred = model(x_test).data.numpy()
    test_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(testing_label_np, 1))) * 100
    print("Testing Set Performance   : " + str(test_perf_i) + "%") 