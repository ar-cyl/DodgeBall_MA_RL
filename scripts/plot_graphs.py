
import matplotlib.pyplot as plt
import numpy as np
import csv
 
X = []
Y = []

with open('purple_returns.txt', 'r') as datafile:
    plotting = csv.reader(datafile)
     
    for ROWS in plotting:
        X.append(float(ROWS[0]))
        
Y = range(len(X))
y_ticks = np.arange(-10, 20, 2)
 

fig, ax = plt.subplots()
plt.plot(Y, X)
plt.yticks(y_ticks)
ax.set_yticklabels(y_ticks)
# fig.autofmt_xdate()

plt.title('purple team returns')
plt.xlabel('Episodes')
plt.ylabel('Avg Returns')
plt.savefig("purple_returns.png")

X = []
Y = []

with open('blue_returns.txt', 'r') as datafile:
    plotting = csv.reader(datafile)
     
    for ROWS in plotting:
        X.append(float(ROWS[0]))
        
Y = range(len(X))
y_ticks = np.arange(-10, 20, 2)
 

fig, ax = plt.subplots()
plt.plot(Y, X)
plt.yticks(y_ticks)
ax.set_yticklabels(y_ticks)
# fig.autofmt_xdate()

plt.title('blue team returns')
plt.xlabel('Episodes')
plt.ylabel('Avg Returns')
plt.savefig("blue_returns.png")


X = []
Y = []

with open('elo_rating', 'r') as datafile:
    plotting = csv.reader(datafile)
     
    for ROWS in plotting:
        X.append(float(ROWS[0]))
        
Y = range(len(X))
# y_ticks = np.arange(-10, 20, 2)
 

fig, ax = plt.subplots()
plt.plot(Y, X)
# plt.yticks(y_ticks)
# ax.set_yticklabels(y_ticks)
# fig.autofmt_xdate()

plt.title('purple team rating')
plt.xlabel('Episodes')
plt.ylabel('Elo rating')
plt.savefig("purple_rating.png")
