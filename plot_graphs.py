import torch
import numpy as np
import matplotlib.pyplot as plt

loadPath = 'SumNet_with_augs/'

N = 60


tr_loss = torch.load(loadPath+'trainLoss.pt')[:N]
val_loss = torch.load(loadPath+'validLoss.pt')[:N]


plt.figure()
plt.plot(tr_loss,'-r',label='train')
plt.plot(val_loss,'-g',label='val')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Plot of loss vs. epochs')
plt.savefig(loadPath+'loss_plot.png')


tr_dice = torch.load(loadPath+'trainDiceCoeff.pt')
tr_dice_class = np.zeros((5,N))
for n in range(N):
	tr_dice_class[:,n] = tr_dice[n]

plt.figure()
plt.plot(tr_dice_class[0],'-r',label='BE')
plt.plot(tr_dice_class[1],'-g',label='Suspicious')
plt.plot(tr_dice_class[2],'-b',label='HGD')
plt.plot(tr_dice_class[3],'-m',label='Cancer')
plt.plot(tr_dice_class[4],'-c',label='Polyp')
plt.legend()
plt.ylabel('Dice')
plt.xlabel('Epochs')
plt.title('Plot of dice vs. epochs')
plt.savefig(loadPath+'train_dice_plot.png')
	


val_dice = torch.load(loadPath+'validDiceCoeff.pt')
val_dice_class = np.zeros((5,N))
for n in range(N):
	val_dice_class[:,n] = val_dice[n]

plt.figure()
plt.plot(val_dice_class[0],'-r',label='BE')
plt.plot(val_dice_class[1],'-g',label='Suspicious')
plt.plot(val_dice_class[2],'-b',label='HGD')
plt.plot(val_dice_class[3],'-m',label='Cancer')
plt.plot(val_dice_class[4],'-c',label='Polyp')
plt.legend()
plt.ylabel('Dice')
plt.xlabel('Epochs')
plt.title('Plot of dice vs. epochs')
plt.savefig(loadPath+'val_dice_plot.png')
