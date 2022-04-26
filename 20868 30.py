import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, cohen_kappa_score, hamming_loss
from sklearn.preprocessing import MinMaxScaler


# Hyper Parameters
EPOCH = 50               # train the training data n times
BATCH_SIZE = 64
TIME_STEP = 48          # rnn time step
INPUT_SIZE = 4         # rnn input size
LR = 0.01               # learning rate
TEST_SIZE = 480
OUTPUT_SIZE = 3
HIDDEN_SIZE = 64


df = pd.read_csv(open(r'...\数据\031-20868.csv')).iloc[:, 1:]

mm = MinMaxScaler()
df[['driving_hour', 'driving_distance']] = mm.fit_transform(df[['driving_hour', 'driving_distance']])

feature = [
    df[['driving_hour', 'driving_distance', 'level', 'daytime']].astype(float).values[i: i + TIME_STEP].tolist()
    for i in range(len(df) - TIME_STEP)
]

label = [
    df['level'].astype(float).values[i + TIME_STEP]
    for i in range(len(df) - TIME_STEP)
]
feature, label = torch.tensor(feature), torch.tensor(label)
train_dataset = TensorDataset(feature[:-TEST_SIZE], label[:-TEST_SIZE])
# Data Loader for easy mini-batch return in training
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

x_test = feature[-TEST_SIZE:]
y_test = label[-TEST_SIZE:]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # bidirectional=True,     # hidden_size要*2
            # dropout=0.2
        )

        self.out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(1, EPOCH+1):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, TIME_STEP, INPUT_SIZE)              # reshape x to (batch, time_step, input_size)
        output = rnn(b_x.to(torch.float32))                               # rnn output
        loss = loss_func(output, b_y.long())                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            val_output = rnn(x_test.to(torch.float32).view(-1, TIME_STEP, INPUT_SIZE))                   # (samples, time_step, input_size)
            val_loss = loss_func(val_output, y_test.long())
            pred_y_val = torch.max(val_output, 1)[1].data.numpy()
            train_acc = accuracy_score(y_test, pred_y_val)
            train_precision = precision_score(y_test, pred_y_val, average='weighted', zero_division=1)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| validation loss: %.4f'
                  % val_loss.data.numpy(), '| train accuracy: %.4f' % train_acc, '| train precision: %.4f' % train_precision)
            
            

test_output = rnn(x_test.to(torch.float32).view(-1, TIME_STEP, INPUT_SIZE))
y_pred = torch.max(test_output, 1)[1].data.numpy()

test_acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
kappa = cohen_kappa_score(y_test, y_pred)
ham_distance = hamming_loss(y_test, y_pred)

print('test accuracy: %.4f' % test_acc)
print('test kappa: %.4f' % kappa)
print('test ham_distance: %.4f' % ham_distance)
print('test precision: %.4f' % precision)



time = []
for i in range(len(y_test)):
    time.append(i)
plt.scatter(time, y_test, marker='o', label='Risk Level')
plt.scatter(time, y_pred+0.1, marker='x', label='Predicted Risk Level')
plt.xlabel('Time')
plt.ylabel('Risk Level')
plt.yticks((0, 1, 2))
plt.legend(loc='center left', bbox_to_anchor=(0, 0.8))
plt.savefig(r'...\20868.tif',dpi=600, format='tif')
plt.show()
