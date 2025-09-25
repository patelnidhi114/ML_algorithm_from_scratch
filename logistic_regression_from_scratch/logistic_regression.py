import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01,num_iterations=1000,lambda_=0.1,verbose=False ):
        self.learning_rate = learning_rate
        self.max_iteration = num_iterations
        self.weights = None
        self.bias = 0
        self.lambda_ = lambda_
        self.verbose = verbose

    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def _compute_cost(self,X,y):
        n = X.shape[0]
        
        z = np.dot(X,self.weights) + self.bias
        yhat = self._sigmoid(z)
        loss_function = (-(y*(np.log(yhat)) + (1-y)*(np.log(1-yhat)))).mean()
        regularization = (self.lambda_/2*n)* np.sum(self.weights**2)
        return loss_function + regularization
    
    def _compute_gradient(self, X,y):
        n = X.shape[0]
        
        z = np.dot(X,self.weights) + self.bias
        yhat = self._sigmoid(z)
        # loss_function = -(y*(np.log(yhat)) + (1-y)*(np.log(1-yhat)))
        # dl/dw = dl/dz * dz/dw
        # dz/dw = x
        # dl/dz = dl/dyhat * dyhat/dz
        # dl/dyhat = -y*1/yhat + 1-y/1-yhat
        # dyhat/dz = -i/u^2 * du/dz = -(-e^-z)/(1+e^-z)^2 = e^z/(1+e^-z)^2
        #                                                 = yhat * (1-yhat)
        # dl/dz = ( -y*1/yhat + 1-y/1-yhat)*(yhat * (1-yhat)) =  (yhat - y)
        # dl/dw = 1/n sigma(yhat - y) * x = 1/n * (XT* (yhat-y))

        # dl/db = dl/dz * dz/db = (yhat - y) = 1/n sigma((yhat - y))
        dw = (1/n) * np.dot(X.T , (yhat-y)) + (self.lambda_/n)*self.weights
        db = (1/n) * np.sum((yhat-y))
        self.weights-= self.learning_rate*dw
        self.bias-= self.learning_rate*db
        

    def fit(self,X,y):
        n,n_features = X.shape
        self.weights = np.zeros(n_features)
        for iteration in range(self.max_iteration):
            self._compute_gradient(X,y)

            if self.verbose and iteration % 200 == 0 : 
                cost = self._compute_cost(X,y)
                print(f" for iteration {iteration} cost : {cost:.4f}")


    def predict_proba(self,X):
        return self._sigmoid(np.dot(X,self.weights)+self.bias)

    def predict(self,X,threshold=0.5):
        activation = self.predict_proba(X)
        return activation >= threshold

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=10, weights=[0.7, 0.3], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(learning_rate=0.1, num_iterations=4200, lambda_=0.1, verbose=True)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
from sklearn.metrics import accuracy_score, f1_score

y_pred = model.predict(X_test_scaled)
# print(y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))