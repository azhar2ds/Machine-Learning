import pandas as pd
import numpy as np
# distanceXconsume
temp_vs_speed = pd.read_csv('https://raw.githubusercontent.com/azhar2ds/Machine-Learning/Linear-Regression/car_fuel_consume.csv',usecols=[3,2],thousands=',').dropna()
car_price_over_time = pd.read_csv('https://raw.githubusercontent.com/azhar2ds/Machine-Learning/Linear-Regression/car_price_prediction.csv',usecols=[3,2]).dropna()

print("Temp VS speed num rows : %d num columns : %d" % temp_vs_speed.shape)
print("Present price VS Selling Price data num rows : %d num columns : %d " % car_price_over_time.shape)


print(len(temp_vs_speed))
print(len(car_price_over_time))

print(temp_vs_speed[0:6],car_price_over_time[0:6])

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

temp,speed = temp_vs_speed.iloc[:, 1:2].values, temp_vs_speed.iloc[:, 0].values
Pres_price,Sel_price = car_price_over_time.iloc[:, 1:2].values, car_price_over_time.iloc[:, 0].values

Pres_priceTrain, Pres_priceTest, Sel_priceTrain, Sel_priceTest = train_test_split(Pres_price,Sel_price, test_size = 1/5)
tempTrain, tempTest, speedTrain, speedTest = train_test_split(temp,speed, test_size = 1/5)
print(" Training data for selling price vs present price num rows : %d num columns : %d" % Pres_priceTrain.shape)
print(" Test data for selling price vs present price num rows     : %d num columns  : %d" % Pres_priceTest.shape)

selPriceModel = LinearRegression()
selPriceModel.fit(Pres_priceTrain, Sel_priceTrain)

car_price_over_time[2:4]

selPriceModel.predict([[9.85],[4.15]])

print(temp_vs_speed[0:6],car_price_over_time[0:6])

#Again, let's train the model and predict
speedModel = LinearRegression()
speedModel.fit(tempTrain, speedTrain)

print(temp_vs_speed.iloc[[0,312]])

print(speedModel.predict([[215],[22]]))

def scatterAnndPlot(x,y,yPred,title,xlabel,ylabel,ax,c):
    ax[c].scatter(x, y, color = 'red')
    ax[c].plot(x, yPred, color = 'blue')
    ax[c].set_title(title)
    ax[c].set_xlabel(xlabel)
    ax[c].set_ylabel(ylabel)
def createSubPlotsEspace(num_rows, num_columns):
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_columns)
    fig.set_size_inches(w=20,h=5)
    return ax
# batch prediction
Sel_pricePredictionTrain = selPriceModel.predict(Pres_priceTrain)
Sel_pricePredictionTest = selPriceModel.predict(Pres_priceTest)
# plot real vs predicted values
PresentVsSellingSpace = createSubPlotsEspace(1,2)
scatterAnndPlot(Pres_priceTrain,Sel_priceTrain,Sel_pricePredictionTrain,
                'Selling price vs Present Price (Training set)',
                'Present price(Million FCFA)',
                'Selling price(Million FCFA)',
                PresentVsSellingSpace,0)
scatterAnndPlot(Pres_priceTest,Sel_priceTest,Sel_pricePredictionTest,
                'Selling price vs Present Price (Test set)',
                'Present price(Million FCFA)',
                'Selling price(Million FCFA)',
                PresentVsSellingSpace,1)
Theta0, Theta1, x = 2,3,4
Theta = np.array([2,3]).reshape(-1,1) #Transform to vector
X = np.array([1,4]).reshape(1,-1)
print("Non vectorized : ", Theta1*x+Theta0)
print("Vectorized     : ", np.matmul(X, Theta))
def addOnesToDataset(X): # make this take unlimited datasets
    N=len(X)
    oneVector = np.ones((N,1))
    return np.hstack((oneVector,X))  
tempTrainWith1, tempTestWith1, Pres_priceTrainWith1, Pres_priceTestWith1 = addOnesToDataset(tempTrain),addOnesToDataset(tempTest), addOnesToDataset(Pres_priceTrain), addOnesToDataset(Pres_priceTest)
def L2(X,Y,prameters):
    N = len(X)
    diffVector = np.matmul(X,prameters.reshape(-1,1)) - Y.reshape(-1,1)
    diffVectorSquare=np.power( diffVector ,2)
    sumTerm = np.sum(diffVectorSquare)
    return sumTerm / (2*N);
a= np.random.rand(5) 
b= np.random.rand(5) 
colors = ['blue','yellow','purple','orange','green']
plt.scatter(Pres_priceTrain, Sel_priceTrain, color = 'red')
for i in range(0,5):
    randomPrediction = Pres_priceTrain*a[i]+b[i]
    plt.plot(Pres_priceTrain,randomPrediction, color = colors[i])
    plt.title("prediction with random a and b ")
    plt.xlabel("Present Price")
    plt.ylabel("Selling Price")
plt.show()
def gradientDescent(X, Y, theta, alpha, epochs): # an epochs is just a fancy term that mean that we have read all the data
    N=len(X) # Number of training examples
    J_over_epochs = np.zeros((epochs+1, 1))  # we will save our erros while training our model 
    theta1_over_epochs = np.zeros((epochs+1, 1)) # we will save values of our theta1 over epochs 
    J_over_epochs[0] = L2(X,Y,theta) # we will also save  initial theta1 and J before runnig gradient descent
    theta1_over_epochs[0] = theta[1]
    for i in range(1,epochs+1):
        ysDiff  = np.matmul(X,theta) - Y.reshape(-1,1) # compute difference between real Y and predicted Y 
        sumForEveryTheta = np.matmul(ysDiff.transpose(),X) # this is a vector of the same size as theta
        gradients = sumForEveryTheta * (1/N) * alpha # element wise multiplication (gradients = sumForEveryTheta * (alpha/m))
        theta = theta-gradients.transpose() # new values of theta 
        theta1_over_epochs[i] = theta[1]
        J_over_epochs[i] = L2(X,Y,theta) 
    return J_over_epochs,theta1_over_epochs, theta
def plotGradientDescent(theta1,J,k,lr,ax,c):
    #global ax
    initial_theta1  = theta1[0]
    J0 = J[0]
    
    ax[c].plot(theta1,J,c='red')

    ax[c].annotate('Starting point', xy=(initial_theta1,J0 ), xytext=(initial_theta1+0.05,J0-1),
            arrowprops=dict(facecolor='black', shrink=0.01),fontsize=15
            )

    ax[c].set_xlabel('theta1',fontsize=20)
    ax[c].set_ylabel('J(error)',fontsize=20)
    ax[c].set_title('L2 Gradient Descent with a %s Learning Rate = %.4f' % (k,lr),fontsize=15)
def plotLossOverEpochs(epochs,J,ax,k,lr,c):
    ax[c].plot(np.arange(0,epochs+1).reshape(-1,1),J,c='red')
    ax[c].set_xlabel('Epochs',fontsize=20)
    ax[c].set_ylabel('J(error)',fontsize=20)
    ax[c].set_title('Loss over epochs with a %s Learning Rate = %.4f' % (k,lr),fontsize=15)
initialTheta = np.zeros((2,1))
epochs = 20
learningRates = {'BIG':0.015,'OKAY':0.001, 'SMALL':0.00005}
thetas = {} # contains final theta for every  learning rate

ThetaVsLossSpace =  createSubPlotsEspace(1,3) # subplots to contain all our three gradient graphic for the different learning rates
LossOverEpochsSpace = createSubPlotsEspace(1,3) # subplots to contain all our three loss over epochs graphic for the different learning rates

for i,(k,lr) in enumerate(learningRates.items()):   
    all_J,theta1_over_epochs,thetas[k] = gradientDescent(Pres_priceTrainWith1,Sel_priceTrain,initialTheta,lr,epochs)  # training a model for the actual learninng rate
    
    plotGradientDescent(theta1_over_epochs,all_J,k,lr,ThetaVsLossSpace,i) # plot gradient descent for every learning rate
    
    plotLossOverEpochs(epochs,all_J,LossOverEpochsSpace,k,lr,i)        # plot loss over epochs for every learning rate
for k,theta in thetas.items():
    print(k,' : ', theta.reshape(-1))
finalTheta = thetas['OKAY']
print("Our Model : Y = %f * x + %f" % tuple(finalTheta)[::-1])
def predict(X, theta):
    return np.matmul(X,theta)
# prediction for training and test set
our_model_sel_price_train_pred = predict(Pres_priceTrainWith1,finalTheta)
our_model_sel_price_test_pred = predict(Pres_priceTestWith1,finalTheta)

# plot training and test prediction vs real values

selPriceVsPresentPriceSpace = createSubPlotsEspace(1,2)
scatterAnndPlot(Pres_priceTrain,Sel_priceTrain,our_model_sel_price_train_pred,
                'Selling price vs Present Price (Training set)',
                'Present price(Million FCFA)',
                'Selling price(Million FCFA)',
                selPriceVsPresentPriceSpace,0
               )
scatterAnndPlot(Pres_priceTest,Sel_priceTest,our_model_sel_price_test_pred,
                'Selling price vs Present Price (Training set)',
                'Present price(Million FCFA)',
                'Selling price(Million FCFA)',
                selPriceVsPresentPriceSpace,1
               )
# give the error for train and test data
from sklearn.metrics import mean_squared_error

def give_me_my_errors(yTrain,yTrainPred,yTest,yTestPred):
    mseTrain = mean_squared_error(yTrain, yTrainPred)
    mseTest = mean_squared_error(yTest, yTestPred)
    return mseTrain, mseTest
Sel_pricePredictionTrain = selPriceModel.predict(Pres_priceTrain)
Sel_pricePredictionTest = selPriceModel.predict(Pres_priceTest)
# compute scikit-learn model errors
mseTrain,mseTest = give_me_my_errors(Sel_priceTrain, Sel_pricePredictionTrain,Sel_priceTest, Sel_pricePredictionTest)

# compute our model errors
trainingError = L2(Pres_priceTrainWith1,Sel_priceTrain,finalTheta)
testError = L2(Pres_priceTestWith1,Sel_priceTest,finalTheta)

print("Sklearn model-Error on Training data :",mseTrain,"\n\tError on Test data:",mseTest)
print("Our Model- : Error on Training data : ", trainingError,"\n\tError on Test data",testError)

# Get predictions
speedPredTrain = speedModel.predict(tempTrain)
speedPredTest = speedModel.predict(tempTest)
# Get erros
mseTrain,mseTest = give_me_my_errors(speedTrain, speedPredTrain,speedTest, speedPredTest)

print("Actual Error and old Problem with (Speed Vs Temperature)")
print(" Error on Training data:",mseTrain,"\n Error on Test data:",mseTest)
speedVsTemPlotSpace = createSubPlotsEspace(1,2)
scatterAnndPlot(tempTrain,speedTrain,speedPredTrain,
                'Speed vs Temperature (Training set)',
                'Temperature(Fahrenheit)','Speed(Miles)',
                   speedVsTemPlotSpace,0)
scatterAnndPlot(tempTest,speedTest,speedPredTest,
                'Speed vs Temperature (Test set)',
                'Temperature(Fahrenheit)','Speed(Miles)',
                   speedVsTemPlotSpace,1) 