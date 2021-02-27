import argparse
import numpy as np
import sys
import datetime


class UserBased:

    def __init__(self, k=10, useRelation=True):
        self.dataDir = '../data/'
        self.K = k
        self.testDataPath = '../data/testing.dat'
        self.useRelation = useRelation
    # Read data from files.

    def readData(self):
        self.normalizedMatrix = np.load(
            self.dataDir + 'Normalized_matrix_customer.npy', allow_pickle=True)
        self.originalMatrix = np.load(
            self.dataDir + 'Original_score_matrix_customer.npy', allow_pickle=True)
        self.movieCount, self.userCount = self.normalizedMatrix.shape
        print("Total User(s): {0}.\nTotal Movie(s): {1}.".format(
            self.movieCount, self.userCount))

        # Record users' mean rating score.
        self.userMean = np.zeros(self.originalMatrix.shape[0])
        for i in range(self.originalMatrix.shape[0]):
            arr = self.originalMatrix[i]
            arr = arr[arr >= 0]
            self.userMean[i] = np.mean(arr)

        self.movieEncodingDict = np.load(
            self.dataDir + 'movie_encoding_dict_customer.npy', allow_pickle=True).item()
        self.userEncodingDict = np.load(
            self.dataDir + 'user_encoding_dict_customer.npy', allow_pickle=True).item()

        # Read test data.
        self.recommendArr = []
        testFile = open(self.testDataPath, mode='r', encoding='utf-8')
        for line in testFile.readlines():
            lineList = line.split(',')
            user = int(lineList[0])
            movie = int(lineList[1])
            user = self.userEncodingDict[user] if user in self.userEncodingDict else -1
            movie = self.movieEncodingDict[movie] if movie in self.movieEncodingDict else -1
            self.recommendArr.append((user, movie))

        self.recommendArr = np.array(self.recommendArr, dtype=[
                                     ('movie', 'i4'), ('user', 'i4')])
        self.sortedIdx = np.argsort(self.recommendArr, order=('movie', 'user'))
        # print(self.recommendArr)
        testFile.close()

        # Read relation data.
        self.relationMatrix = np.load(
            self.dataDir + 'relation_matrix_customer.npy', allow_pickle=True)
        self.factorMatrix = np.zeros(self.relationMatrix.shape)
        for i in range(self.relationMatrix.shape[0]):
            for j in range(self.relationMatrix.shape[1]):
                if i == j:
                    self.factorMatrix[i][j] = 1.0
                else:
                    if self.relationMatrix[i][j] == 1 and self.relationMatrix[j][i] == 1:
                        self.factorMatrix[i][j] = 1.0
                    elif self.relationMatrix[i][j] == 0 and self.relationMatrix[j][i] == 0:
                        self.factorMatrix[i][j] = 0.6
                    else:
                        self.factorMatrix[i][j] = 0.8

        timeNow = datetime.datetime.now()
        self.outputFile = open(self.dataDir + "output_{:02}{:02}{:02}_{:02}{:02}{:02}_{:03}_k_{}_useRelation_{}.txt".format(timeNow.year, timeNow.month, timeNow.day,
                                                                                                       timeNow.hour, timeNow.minute, timeNow.second, timeNow.microsecond, self.K, self.useRelation), mode="w", encoding='utf-8')

    # Calculate simularity between vector a & vector b.
    def simularity(self, a, b):

        if np.dot(a, a) == 0 or np.dot(b, b) == 0:
            return 0
        return np.dot(a, b)/np.sqrt(np.dot(a, a) * np.dot(b, b))

    # Recommender using item-based method.
    def UserBasedRecommend(self):

        result = []
        # init last movie = -1
        lastMovieIdx = -1

        for index in self.sortedIdx:
            movieIdx, userIdx = self.recommendArr[index]

            # print("(Movie Idx, User Idx): ({0}, {1}).".format(movieIdx, userIdx))

            if movieIdx == -1:
                predGrade = 3  # Or maybe random value for this
                result.append(index, predGrade)
                # print("(User, Movie): ({0}, {1}). Result: {2}".format(
                #     movieIdx, userIdx, predGrade))
                continue

            if movieIdx != lastMovieIdx:
                lastMovieIdx = movieIdx
                # Record Mean Value
                mean = self.userMean[movieIdx]

                # Calculate simularity.
                print("\n\nMovie Idx:", movieIdx)
                print(self.normalizedMatrix[movieIdx].shape)

                # normalized vector != zero vector
                if np.dot(self.normalizedMatrix[movieIdx], self.normalizedMatrix[movieIdx]) != 0:

                    simVec = np.zeros(self.movieCount, dtype=np.float32)
                    for movieIdx2 in range(self.movieCount):
                        simVec[movieIdx2] = self.simularity(
                            self.normalizedMatrix[movieIdx], self.normalizedMatrix[movieIdx2])

                    # Using Factor
                    if self.useRelation:
                        simVec = self.factorMatrix[movieIdx] * simVec

                    # np.argsort simularities.
                    # Sort from large to small.
                    simOrder = np.argsort(-simVec)
                    # Only consider sim >= 0.
                    simOrder = simOrder[:np.sum(simVec >= 0)]
                    print(simVec, simOrder)

            if np.dot(self.normalizedMatrix[movieIdx], self.normalizedMatrix[movieIdx]) == 0:
                predGrade = int(round(mean))
                result.append((index, predGrade))
                # print("(User, Movie): ({0}, {1}). Result: {2}".format(
                #     movieIdx, userIdx, predGrade))
                continue

            if self.originalMatrix[movieIdx][userIdx] == -1:

                # Find k neighbors
                neighbors = []
                for movieIdx2 in simOrder:
                    if len(neighbors) >= self.K:
                        break
                    if movieIdx != movieIdx2 and self.originalMatrix[movieIdx2][userIdx] != -1:
                        neighbors.append(movieIdx2)

                # Calculate predict grade.
                predGrade, simSum = 0., 0.
                if len(neighbors) > 0:
                    for neighbor in neighbors:
                        simSum += simVec[neighbor]
                        predGrade += simVec[neighbor] * (
                            self.originalMatrix[neighbor][userIdx] - self.userMean[neighbor])
                    if simSum == 0:
                        predGrade = mean
                    else:
                        predGrade /= simSum
                        predGrade += mean
                else:
                    predGrade = mean
                # print(mean, predGrade, simSum)
                predGrade = int(round(predGrade))
                if predGrade > 5:
                    predGrade = 5
                if predGrade < 0:
                    predGrade = 0

            else:
                predGrade = self.originalMatrix[movieIdx][userIdx]

            # print("(User, Movie): ({0}, {1}). Result: {2}".format(
            #     movieIdx, userIdx, predGrade))
            result.append((index, predGrade))

        return sorted(result)

    def debugOutput(self):
        pass

    def saveRecommendMatrix(self):
        np.save(self.dataDir + 'User_based_recommend_matrix.npy',
                self.originalMatrix)

    def writeToFile(self, result):
        for idx, grade in result:
            self.outputFile.write("{0}\n".format(grade))
        self.outputFile.close()

    def run(self):
        self.readData()
        # self.debugOutput()
        self.result = self.UserBasedRecommend()
        print(self.result)
        self.writeToFile(self.result)

    def debugRun(self, k=2, useRelation=False):
        self.useRelation = useRelation
        self.K = k
        print("Reading Data...")
        self.readData()
        print("Recommend Arr", self.recommendArr, "Original Matrix", self.originalMatrix, "Normalized Matrix", self.normalizedMatrix, "User Mean Matrix",
              self.userMean, "Relation Matrix:", self.relationMatrix, "Relation Factor Matrix:", self.factorMatrix, sep='\n')
        self.result = self.UserBasedRecommend()
        print(self.result)
        self.writeToFile(self.result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=10, required=False,
                        help="specify the value K for K nearest neighbors")
    parser.add_argument('--relation', action="store_true",
                        required=False, help="specify whether to use relation or not")
    args = parser.parse_args()
    if 'k' not in args:
        recommender = UserBased(useRelation=args.relation)
    else:
        recommender = UserBased(args.k, args.relation)
    print(recommender.K, recommender.useRelation)
    recommender.run()
