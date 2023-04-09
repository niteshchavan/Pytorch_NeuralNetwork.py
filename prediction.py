import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import mysql.connector

# Establish database connection
cnx = mysql.connector.connect(user='nitesh', password='root@123',
                              host='localhost', database='market')
cursor = cnx.cursor()

# Get list of tables in database
query = "SHOW TABLES"
cursor.execute(query)

tables = [table[0] for table in cursor]

# Loop through tables and check if column1 and column2 values are less than 30
for table in tables:
    query = f"SELECT Date, Open, High, Low, Close FROM {table};"
    cursor.execute(query)
    results = cursor.fetchall()

    # Close cursor and database connection
    #cursor.close()
    #cnx.close()

    # Load the data into a Pandas dataframe
    df = pd.DataFrame(results, columns=['Date', 'Open', 'High', 'Low', 'Close'])

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
    print(f"Predicted closing price for {table} on {next_date_str}: Rs {next_pred[0][0]:.2f}")
