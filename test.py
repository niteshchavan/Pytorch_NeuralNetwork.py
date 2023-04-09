import mysql.connector
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Connect to the MySQL database
mydb = mysql.connector.connect(
  host="localhost",
  user="nitesh",
  password="root@123",
  database="market"
)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 16) # input layer (4 nodes) -> hidden layer (16 nodes)
        self.fc2 = nn.Linear(16, 4) # hidden layer (16 nodes) -> output layer (3 nodes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load and preprocess the dataset
cursor = mydb.cursor()
cursor.execute("SELECT open, high, low, close FROM ITC")
rows = cursor.fetchall()

# Convert the data to a PyTorch tensor
data = torch.tensor(rows, dtype=torch.float32)

# Normalize the data
data = (data - torch.mean(data, dim=0)) / torch.std(data, dim=0)

# Split the dataset into inputs and labels
inputs = data[:-1]
labels = data[1:]

# Create the neural network and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Train the neural network
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Print the training loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


# Predict the next day's closing price
with torch.no_grad():
    last_row = inputs[-1].unsqueeze(0)
    predicted = net(last_row)
#    print('Predicted closing price: ${:.2f}'.format(predicted.item()))

    predicted = (predicted * torch.std(data, dim=0)[-1]) + torch.mean(data, dim=0)[-1]
#    print('Predicted closing price: ${:.2f}'.format(predicted[0]))
    print(predicted)
