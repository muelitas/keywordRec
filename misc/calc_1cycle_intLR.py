'''/**************************************************************************
    File: calc_1cycle_intLR.py
    Author: Mario Esparza
    Date: 03/14/2021
    
    Given the graph from find_LR, provide this program with optimal Learning
    Rate and with number of epochs that will be used in the run. The program
    will return a good estimate for the 'max_lr' attribute of the 1cycle
    learning rate scheduler.
    
***************************************************************************''' 

epochs = 100
optimal_LR = 9.5e-6
max_lar = -1

#Next, given the number of epochs, find the right ratio
if epochs == 100:
    max_lr = 18 * optimal_LR
elif epochs == 90:
    max_lr = 17.36 * optimal_LR
elif epochs == 80:
    max_lr = 16.72 * optimal_LR
elif epochs == 70:
    max_lr = 15.97 * optimal_LR
elif epochs == 60:
    max_lr = 15.06 * optimal_LR
elif epochs == 50:
    max_lr = 14 * optimal_LR
elif epochs == 40:
    max_lr = 12.56 * optimal_LR
elif epochs == 30:
    max_lr = 10.77 * optimal_LR
elif epochs == 20:
    max_lr = 8.38 * optimal_LR
else:
    print("I am sorry, but I don't have the ratio registered for this number"
          " '{epochs}' of epochs.")

print(f"The values you should set up in your run are:\n\tLR: {max_lr:e}\n\t"
      f"LR: {max_lr:.8f}\n\tEpochs: {epochs}")