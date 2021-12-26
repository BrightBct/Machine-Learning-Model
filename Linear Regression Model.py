# noinspection PyPep8Naming,PyUnusedLocal,SpellCheckingInspection,PyTypeChecker
class LinearRegressionModel:
    def __init__(self, epochs=1000, learning_rate=-99, method=0, batch_size=0):
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__method = method
        self.__batch_size = batch_size
        self.copy = __import__('copy')
        self.math = __import__('math')
        self.random = __import__('random')
        self.__X = None
        self.__Y = None
        self.__test = None
        self.__theta = None
        self.__se = None
        self.__J_History = []
        self.__rmse = None

    def find_hypothesis(self, theta, row):
        return sum([theta[col] * self.__X[row][col] for col in range(len(self.__X[0]))])

    def find_J_of_theta(self, theta):
        JofTheta = 0
        for row in range(len(self.__X)):
            hypothesis = self.find_hypothesis(theta, row)
            JofTheta += self.math.pow(hypothesis - self.__Y[row], 2)
        JofTheta /= 2
        self.__J_History.append(JofTheta)

    def find_batch_gradient(self, theta, col):
        gradient = 0
        for row in range(len(self.__X)):
            hypothesis = self.find_hypothesis(theta, row)
            gradient += (hypothesis - self.__Y[row]) * self.__X[row][col]
        return gradient

    def find_stochastic_gradient(self, theta, col, random_value):
        hypothesis = self.find_hypothesis(theta, random_value)
        gradient = (hypothesis - self.__Y[random_value]) * self.__X[random_value][col]
        return gradient

    def find_mini_batch_gradient(self, theta, col, random_value):
        gradient = 0
        for row in random_value:
            hypothesis = self.find_hypothesis(theta, row)
            gradient += (hypothesis - self.__Y[row]) * self.__X[row][col]
        return gradient

    def find_best_learning_rate(self, theta):
        best_learning_rate = False
        if self.__J_History[0] <= self.__J_History[1]:
            self.__learning_rate /= 1.1
            theta = [0 for _ in range(len(self.__X[0]))]
            self.__J_History.clear()
            self.find_J_of_theta(theta)
        elif self.__J_History[0] > self.__J_History[1]:
            best_learning_rate = True
            self.__theta = theta
        return zip(best_learning_rate, theta)

    def find_best_theta(self):
        theta = [0 for _ in range(len(self.__X[0]))]
        self.find_J_of_theta(theta)

        best_learning_rate = False
        if self.__learning_rate == -99:
            self.__learning_rate = 1
            best_learning_rate = False
        else:
            best_learning_rate = True

        iteration = 0

        while True:
            if self.__method == 0:
                new_theta = [round(theta[col] - (self.__learning_rate * self.find_batch_gradient(theta, col)), 10)
                             for col in range(len(self.__X[0]))]
                theta = new_theta.copy()

            elif self.__method == 1:
                random_value = self.random.randint(0, len(self.__X) - 1)
                new_theta = [round(theta[col] -
                                   (self.__learning_rate * self.find_stochastic_gradient(theta, col, random_value)), 10)
                             for col in range(len(self.__X[0]))]
                theta = new_theta.copy()

            elif self.__method == 2:
                random_value = [self.random.randint(0, len(self.__X) - 1) for _ in range(self.__batch_size)]
                new_theta = [round(theta[col] -
                                   (self.__learning_rate * (1 / self.__batch_size) *
                                    self.find_mini_batch_gradient(theta, col, random_value)), 10)
                             for col in range(len(self.__X[0]))]
                theta = new_theta.copy()

            self.find_J_of_theta(theta)

            if best_learning_rate:
                predict = self.predict(self.__X)
                self.find_square_error(theta, predict)
                self.find_root_mean_square_error(theta, predict)
                iteration += 1
            else:
                best_learning_rate, theta = self.find_best_learning_rate(theta)

            if iteration == self.__epochs:
                break

    def find_root_mean_square_error(self, theta, predict):
        rmse_value = 0
        for i in range(len(predict)):
            rmse_value += self.math.pow((predict[i] - self.__Y[i]), 2)
        rmse_value = self.math.sqrt(rmse_value / len(predict))
        self.__rmse.append(rmse_value)

    def find_square_error(self, theta, predict):
        self.__se.append(sum([self.math.pow(predict[i] - self.__Y[i], 2) for i in range(len(predict))]))

    def fit(self, X, Y):
        self.__X = self.copy.deepcopy(X)
        self.__Y = self.copy.deepcopy(Y)

        for i in range(len(self.__X)):
            self.__X[i].append(1)

        self.find_best_theta()

    def predict(self, test):
        self.__test = self.copy.deepcopy(test)
        predict = []
        if self.__X != self.__test:
            for i in range(len(self.__test)):
                self.__test[i].append(1)

        for row in range(len(self.__test)):
            hypothesis = self.find_hypothesis(self.__theta, row)
            predict.append(round(hypothesis))
        return predict

    def getTheta(self):
        return self.__theta

    def getSquareError(self):
        return self.__se

    def getJofTheta(self):
        return self.__J_History

    def getRootMeanSqureError(self):
        return self.__rmse

    def getEpochs(self):
        return self.__epochs

    def getLearningRate(self):
        return self.__learning_rate

    def getGradientMethod(self):
        if self.__method == 0:
            return "Batch Gradient Descent"
        elif self.__method == 1:
            return "Stochastic Gradient Descent"
        elif self.__method == 2:
            return "Mini-Batch Gradient Descent"

    def getBatchSize(self):
        return self.__batch_size
