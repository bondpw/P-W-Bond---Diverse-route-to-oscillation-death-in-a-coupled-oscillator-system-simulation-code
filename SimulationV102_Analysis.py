import numpy as np
import matplotlib.pyplot as plt

x1Data = np.load("x1Data.npy",allow_pickle = True)
x2Data = np.load("x2Data.npy", allow_pickle = True)
x3Data = np.load("x3Data.npy", allow_pickle = True)
timeData = np.load("timedata.npy", allow_pickle = True)

plt.plot(timeData, x1Data, label = "x1(t)")
plt.plot(timeData, x2Data, label = "x2(t)")
plt.plot(timeData, x3Data, label = "x3(t)")
plt.title("Time Series")
plt.xlabel("Time/s")
plt.ylabel("Component Position")
plt.legend()
plt.show()

plt.plot(timeData, x1Data)
plt.xlabel("Time/s")
plt.ylabel("Amplitude")
plt.title("Cardiac Component")
plt.show()

plt.plot(timeData, x2Data)
plt.xlabel("Time/s")
plt.ylabel("Amplitude")
plt.title("Respiratory Component")
plt.show()

plt.plot(timeData, x3Data)
plt.xlabel("Time/s")
plt.ylabel("Amplitude")
plt.title("Myogenic Component")
plt.show()

