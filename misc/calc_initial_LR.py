'''/**************************************************************************
    File: calc_initial_LR.py
    Author: Mario Esparza
    Date: 03/14/2021
    
    Given the graph from find_LR, and depending on the Learning Rate Scheduler
    that is to be used; provide optimal Learning Rate, number of epochs, value
    for Gamma, etc. The program will return a good estimate for the LR to use.
    
***************************************************************************''' 
#Use this if you are thinking of using Exponential LR Scheduler
init_LR = 1e-4
gamma = 0.982
epochs = 50
end_LR = init_LR

for i in range(1, epochs+1):
    end_LR *= gamma
    
print(f"After {epochs} epochs, starting with {init_LR}, your ending LR would"
      f" be {end_LR}")

#Use this if you are thinking of using oneCycle LR Scheduler
# epochs = 100
# optimal_LR = 9.5e-6
# max_lar = -1

# #Next, given the number of epochs, find the right ratio
# if epochs == 100:
#     max_lr = 18 * optimal_LR
# elif epochs == 90:
#     max_lr = 17.36 * optimal_LR
# elif epochs == 80:
#     max_lr = 16.72 * optimal_LR
# elif epochs == 70:
#     max_lr = 15.97 * optimal_LR
# elif epochs == 60:
#     max_lr = 15.06 * optimal_LR
# elif epochs == 50:
#     max_lr = 14 * optimal_LR
# elif epochs == 40:
#     max_lr = 12.56 * optimal_LR
# elif epochs == 30:
#     max_lr = 10.77 * optimal_LR
# elif epochs == 20:
#     max_lr = 8.38 * optimal_LR
# else:
#     print("I am sorry, but I don't have the ratio registered for this number"
#           " '{epochs}' of epochs.")

# print(f"The values you should set up in your run are:\n\tLR: {max_lr:e}\n\t"
#       f"LR: {max_lr:.8f}\n\tEpochs: {epochs}")
