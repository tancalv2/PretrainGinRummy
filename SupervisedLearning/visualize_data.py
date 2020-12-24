import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def visualizeClasses(plot_pth, actions, classes):
    actions_count = np.sum(actions, axis=0)
    plt.figure(figsize=(30,30))
    plt.title("Class Distribution", fontsize=14)
    plt.xlabel("Actions", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.bar(classes, actions_count, log=True)
    plt.savefig('{}/action_count.png'.format(plot_pth),bbox_inches='tight')
    plt.show()
    plt.close()


def plotTrain(plot_pth, train, val, epoch, label, bs, lr):
    plt.figure()
    plt.title("{} vs. Epoch".format(label), fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.plot(train, label='Training {}'.format(label))
    plt.plot(val, label='Validation {}'.format(label))
    plt.axvline(epoch, color='red', label='Best Epoch: {}'.format(epoch+1))
    plt.plot(epoch, val[epoch], marker='o', color="red", label="{}: {:.4f}"
             .format(label, val[epoch]))
    plt.legend()
    plt.savefig('{}/{}_bs_{}_lr_{}.png'.format(plot_pth,label,bs,lr),bbox_inches='tight')
    plt.show()
    plt.close()


def evaluate_confusion_matrix(model, data_loader, device):
    """
    Run the model on the test set and generate the confusion matrix.

    Args:
        model: PyTorch neural network object
        data_loader: PyTorch data loader for the dataset
    Returns:
        cm: A NumPy array denoting the confusion matrix
    """
    val_labels = np.array([], dtype=np.int64)
    val_preds = np.array([], dtype=np.int64)

    for i, data in enumerate(data_loader, 0):
        vinputs, vlabels = data
        vinputs = vinputs.type(torch.FloatTensor).to(device)
        vlabels = vlabels.type(torch.FloatTensor)
        voutputs = model(vinputs)
        vguess = torch.argmax(voutputs.cpu(), dim=1)
        vlabels = torch.argmax(vlabels.cpu(), dim=1)
        val_labels = np.concatenate((val_labels, vlabels))
        val_preds = np.concatenate((val_preds, vguess))
    
    # cm = confusion_matrix(val_labels, val_preds)
    cm_temp = np.zeros([voutputs.shape[1],voutputs.shape[1]], dtype=np.int64)
    for i in range(len(val_labels)):
        cm_temp[val_labels[i]][val_preds[i]] += 1
    cm = cm_temp
    print('Accuracy: {:.2f}'.format(100*(val_labels == val_preds).sum() /len(val_labels)))
    return cm


# Function based off
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(plot_pth, cm, classes, numGames, mode,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        cm: A NumPy array denoting the confusion matrix
        classes: A list of strings denoting the name of the classes
        normalize: Boolean whether to normalize the confusion matrix or not
        title: String for the title of the plot
        cmap: Colour map for the plot
    """

    # plot name
    plot_name = '{}/CM_G_{}k'.format(plot_pth,numGames//1000)
    plot_name = '{}_{}'.format(plot_name,mode) if mode != 'full' else plot_name

    # normalize
    if normalize:
        plot_name = '{}_norm.png'.format(plot_name)
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-5)
        print("Normalized confusion matrix")
    else:
        plot_name = '{}.png'.format(plot_name)
        print('Confusion matrix, without normalization')

    # limit figure size
    if cm.shape[0] < 52:
        plt.figure(facecolor='white')
    else:
        plt.figure(figsize=cm.shape, facecolor='white')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
           
    plt.savefig(plot_name,bbox_inches='tight')
    if cm.shape[0] < 53:
        plt.show()
    plt.close()
    return


def plot_cm(plot_pth, classes, model, data_loader, device, numGames, mode='full'):

    cm = evaluate_confusion_matrix(model, data_loader, device)
    plot_confusion_matrix(plot_pth, cm, classes, numGames, mode,
                        normalize=True,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues)
    plot_confusion_matrix(plot_pth, cm, classes, numGames, mode,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues)