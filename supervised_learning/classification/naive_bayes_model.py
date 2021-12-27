# noinspection PyPep8Naming
class NaiveBayesModel:
    def __init__(self):
        self.np = __import__('numpy')
        self.pd = __import__('pandas')
        self.math = __import__('math')
        self.__dict_output = None
        self.__count = None
        self.__check_value_type = None
        self.__model = None
        
    def create_dictionary_output(self, Y):
        dict_output = {}
        count = 0
        for i in Y:
            if i not in dict_output:
                dict_output[i] = 1
            else:
                dict_output[i] += 1
            count += 1

        self.__dict_output = dict_output
        self.__count = count

    def find_value_type(self, X):
        threshold = round(len(X) * 0.01)

        check_value_type = []
        for i in range(len(X[0])):
            unique_list = []
            for j in range(len(X)):
                if X[j][i] not in unique_list:
                    unique_list.append(X[j][i])
            if len(unique_list) > threshold:
                check_value_type.append(True)
            else:
                check_value_type.append(False)

        self.__check_value_type = check_value_type

    def fit(self, X, Y):

        self.create_dictionary_output(Y)
        self.find_value_type(X)

        model = []
        for i in range(len(self.__check_value_type)):
            if self.__check_value_type[i]:
                dict_feature = {}
                for j in range(len(X)):
                    if Y[j] not in dict_feature:
                        dict_feature[Y[j]] = [X[j][i]]
                    else:
                        dict_feature[Y[j]].append(X[j][i])

                for j in dict_feature:
                    array = self.np.array(dict_feature[j])
                    dict_feature[j] = [self.np.mean(array), self.np.std(array)]
                model.append(dict_feature)

            else:
                dict_feature = self.pd.DataFrame(columns=self.__dict_output.keys())
                data = [[0 for _ in range(len(self.__dict_output))]]
                for j in range(len(X)):
                    if X[j][i] not in dict_feature.index:
                        dict_feature = dict_feature.append(
                            self.pd.DataFrame(data, columns=self.__dict_output.keys(), index=[X[j][i]]))
                    dict_feature.loc[X[j][i]][Y[j]] += 1
                model.append(dict_feature)
        self.__model = model

    def predict(self, test):
        predict = []
        for i in range(len(test)):
            list_sum = {}
            for j in self.__dict_output:
                list_sum[j] = 1
                for k in range(len(self.__check_value_type)):
                    if self.__check_value_type[k]:
                        cal = (1 / (self.math.sqrt(2 * self.math.pi) * self.__model[k][j][1])) * (1 / (self.math.e ** (
                                    ((test[i][k] - self.__model[k][j][0]) ** 2) / (2 * (self.__model[k][j][1] ** 2)))))
                        list_sum[j] *= cal
                    else:
                        list_sum[j] *= self.__model[k].loc[test[i][k], j] / self.__dict_output[j]
                list_sum[j] *= self.__dict_output[j] / self.__count
            predict.append(max(list_sum, key=list_sum.get))
        return predict

    def getModel(self):
        return self.__model
