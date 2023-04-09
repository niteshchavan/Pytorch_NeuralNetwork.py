import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import mysql.connector

# Connect to the MySQL database
mydb = mysql.connector.connect(
  host="localhost",
  user="nitesh",
  password="root@123",
  database="market"
)

# Select the date, open, high, low, and close columns from the ITC table
sql = "SELECT Date, Open, High, Low, Close FROM ITC"

# Load the data into a Pandas dataframe
df = pd.read_sql_query(sql, mydb)

# Extract the closing price
close = df[['Close']].values

# Normalize the data
scaler = MinMaxScaler()
close = scaler.fit_transform(close)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the model, loss function, and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Define the training data
x_train = torch.tensor(close[:-1], dtype=torch.float32)
y_train = torch.tensor(close[1:], dtype=torch.float32)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    output = net(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# Make a prediction on the last row of the input data
last_row = torch.tensor(close[-1], dtype=torch.float32).view(1, -1)
predicted = scaler.inverse_transform(net(last_row).detach().numpy())

# Get the date of the last row of the input data
last_date = df.iloc[-1]['Date']

# Make a prediction for the next day
next_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
next_date_str = next_date.strftime('%Y-%m-%d')
next_row = torch.tensor(close[-1], dtype=torch.float32).view(1, -1)
next_pred = scaler.inverse_transform(net(next_row).detach().numpy())

# Print the predicted closing price with the date
print('Predicted closing price for {}: Rs{:.2f}'.format(next_date_str, next_pred[0][0]))
