import numpy as np


class ItemBased:
    def __init__(self, k=10):
        self.dataDir = '../testingdata/'
        self.K = k
        self.testDataPath = '../testingdata/testing.dat'

    # Read data from files.
    def readData(self):
        self.normalizedMatrix = np.load(
            self.dataDir + 'Normalized_matrix.npy', allow_pickle=True)
        self.originalMatrix = np.load(
            self.dataDir + 'Original_score_matrix.npy', allow_pickle=True)
        self.movieCount, self.userCount = self.normalizedMatrix.shape
        print("Total Movie(s): {0}.\nTotal User(s): {1}.".format(
            self.movieCount, self.userCount))

        self.movieEncodingDict = np.load(
            self.dataDir + 'movie_encoding_dict.npy', allow_pickle=True).item()
        self.userEncodingDict = np.load(
            self.dataDir + 'user_encoding_dict.npy', allow_pickle=True).item()

        # Read test data.
        self.recommendArr = []
        testFile = open(self.testDataPath, mode='r', encoding='utf-8')
        for line in testFile.readlines():
            lineList = line.split(',')
            user = int(lineList[0])
            movie = int(lineList[1])
            user = self.userEncodingDict[user] if user in self.userEncodingDict else -1
            movie =  self.movieEncodingDict[movie] if movie in self.movieEncodingDict else -1
            self.recommendArr.append((movie, user))

        self.recommendArr = np.array(self.recommendArr, dtype=[('movie', 'i4'), ('user', 'i4')])
        self.sortedIdx = np.argsort(self.recommendArr, order=('movie', 'user'))
        # print(self.recommendArr)
        testFile.close()

        self.outputFile = open(self.dataDir + "output.txt", mode="w", encoding='utf-8')

    # Calculate simularity between vector a & vector b.
    def simularity(self, a, b):

        if np.dot(a, a) == 0 or np.dot(b, b) == 0:
            return 0
        return np.dot(a, b)/np.sqrt(np.dot(a, a) * np.dot(b, b))

    # Recommender using item-based method.
    def ItemBasedRecommend(self):

        result = []
        # init last movie = -1
        lastMovieIdx = -1

        for index in self.sortedIdx:
            movieIdx, userIdx = self.recommendArr[index]

            # print("(Movie Idx, User Idx): ({0}, {1}).".format(movieIdx, userIdx))

            if movieIdx == -1:
                predGrade = 3  # Or maybe random value for this
                result.append(index, predGrade)  
                print("(Movie, User): ({0}, {1}). Result: {2}".format(
                    movieIdx, userIdx, predGrade))
                continue

            if movieIdx != lastMovieIdx:
                lastMovieIdx = movieIdx
                # Record Mean Value
                gradeArray = self.originalMatrix[movieIdx]
                mean = np.mean(gradeArray[gradeArray >= 0])

                # Calculate simularity.
                print("\n\nMovie Idx:", movieIdx)
                print(self.normalizedMatrix[movieIdx].shape)

                # normalized vector != zero vector
                if np.dot(self.normalizedMatrix[movieIdx], self.normalizedMatrix[movieIdx]) != 0:

                    simVec = np.zeros(self.movieCount, dtype=np.float32)
                    for movieIdx2 in range(self.movieCount):
                        simVec[movieIdx2] = self.simularity(
                            self.normalizedMatrix[movieIdx], self.normalizedMatrix[movieIdx2])

                    # np.argsort simularities.
                    # Sort from large to small.
                    simOrder = np.argsort(-simVec)
                    # Only consider sim >= 0.
                    simOrder = simOrder[:np.sum(simVec >= 0)]
                    print(simVec, simOrder)
            
            if np.dot(self.normalizedMatrix[movieIdx], self.normalizedMatrix[movieIdx]) == 0:
                predGrade = int(round(mean))
                result.append((index, predGrade))
                print("(Movie, User): ({0}, {1}). Result: {2}".format(
                    movieIdx, userIdx, predGrade))
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
                        predGrade += simVec[neighbor] * \
                            self.originalMatrix[neighbor][userIdx]
                    predGrade /= simSum
                else:
                    predGrade = mean
                predGrade = int(round(predGrade))

            else:
                predGrade = self.originalMatrix[movieIdx][userIdx]

            print("(Movie, User): ({0}, {1}). Result: {2}".format(
                movieIdx, userIdx, predGrade))
            result.append((index, predGrade))

        return sorted(result)

    def debugOutput(self):
        pass

    def saveRecommendMatrix(self):
        np.save(self.dataDir + 'Item_based_recommend_matrix.npy',
                self.originalMatrix)

    def writeToFile(self, result):
        for idx, grade in result:
            self.outputFile.write("{0}\n".format(grade))
        self.outputFile.close()

    def run(self):
        self.readData()
        # self.debugOutput()
        res = self.ItemBasedRecommend()
        print(res)
        self.writeToFile(res)


if __name__ == '__main__':
    recommender = ItemBased(10)
    recommender.run()
