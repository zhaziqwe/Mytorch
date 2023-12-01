# this is lsx's text file 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as sk_f1_score
import cupy as cp

class Performance():

    def __init__(self):
        self.acc = []
        self.loss = []
        self.label = []
        self.pred = []

    def update_in_train(self,acc,loss):
        '''
        update the data in train function to draw accuracy and loss curve
        label : the true y in dataset
        pred  : the model's prediction
        '''
        if isinstance(acc, cp.ndarray):
        # convert cupy to numpy
            acc = cp.asnumpy(acc)
            loss = cp.asnumpy(loss)

        self.acc.append(acc)
        self.loss.append(loss)

    def update_in_test(self,label,pred):
        '''
        update the data in test function to draw matrix and calculate f1-score
        label : the true y in dataset
        pred  : the model's prediction
        '''
        if isinstance(label, cp.ndarray):
        # convert cupy to numpy
            label = cp.asnumpy(label)
            pred = cp.asnumpy(pred)
        self.label.extend(label)
        self.pred.extend(pred)
        
    def reset(self):
        self.__init__()

    def __str__(self):
        return f"accuarcy\t loss\t \n{self.acc[-1]}\t\t {self.loss[-1]}\t"
    
    def graph(self, W = 8 , H = 5):
        '''
        draw Loss and Acc curve
        W : the width of the figure
        H : the height of the figure
        '''
        plt.figure(figsize=(W, H))
        epochs = range(1, len(self.acc) + 1)
        plt.title('Accuracy and Loss')
        plt.xlabel('Epochs')
        plt.plot(epochs, self.acc, 'red', label='Training acc')
        plt.plot(epochs, self.loss, 'blue', label='Validation loss')
        plt.legend()
        plt.show()

    def matrix(self, size = 10):
        '''
        draw confusion matrix
        size : figure size is [size * size] , for 10 * 10 matrix, we suggest [size > 10]
        '''
        plt.figure(figsize=(size,size))
        cm = confusion_matrix(self.label,self.pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')  # 调整颜色等参数
        plt.show()

    def f1_score(self,show_now = True):
        '''
        if show_now is true function will print f1-score immediately 
        otherwise function just return f1-score
        '''
        f1 = sk_f1_score(self.label, self.pred , average='weighted')
        if show_now:
            print(f"f1-score : {f1}" )
        return f1