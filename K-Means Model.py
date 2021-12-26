class KMeansModel:
    def __init__(self, n_clusters=3, max_iter=1000):
        self.__random = __import__('random')
        self.__math = __import__('math')
        self.__n_clusters = n_clusters
        self.__max_iter = max_iter
        self.__df = None
        self.__centroid = None

    def find_predict_value(self, df, centroid):
        predict_value = []
        for row in range(len(df)):
            euclidean_distance = [self.__math.sqrt(sum(self.__math.pow(centroid[k][col] - df.iloc[row, col], 2)
                                                       for col in range(len(df.columns))))
                                  for k in range(self.__n_clusters)]
            predict_value.append(euclidean_distance.index(min(euclidean_distance)))
        return predict_value

    def fit(self, df):
        self.__df = df

    def predict(self):
        centroid = [self.__df.iloc[self.__random.randint(0, len(self.__df) - 1)].values.tolist() for _ in range(self.__n_clusters)]
        predict_value = self.find_predict_value(self.__df, centroid)

        for _ in range(self.__max_iter):
            count_value_cluster = {}
            for predict in predict_value:
                if predict not in count_value_cluster:
                    count_value_cluster[predict] = 0
                else:
                    count_value_cluster[predict] += 1

            mean_cluster = [[0 for _ in range(len(self.__df.columns))] for _ in range(self.__n_clusters)]
            for row in range(len(self.__df)):
                for col in range(len(self.__df.columns)):
                    mean_cluster[predict_value[row]][col] += self.__df.iloc[row, col]
            for k in range(self.__n_clusters):
                centroid[k] = [mean_cluster[k][col] / count_value_cluster[k] for col in range(len(self.__df.columns))]

            predict_value = self.find_predict_value(self.__df, centroid)

        self.__centroid = centroid
        return predict_value

    def get_centroid(self):
        return self.__centroid
