import numpy as np

class ItemBased:
    def __init__(self, k = 10):
        self.dataDir = '../testingdata/'
        self.K = k

    # Read data from files.
    def readData(self):
        self.normalizedMatrix = np.load(self.dataDir + 'Normalized_matrix.npy', allow_pickle=True)
        self.originalMatrix = np.load(
            self.dataDir + 'Original_score_matrix.npy', allow_pickle=True)
        self.movieCount, self.userCount = self.normalizedMatrix.shape
        print("Total Movie(s): {0}.\nTotal User(s): {1}.".format(self.movieCount, self.userCount))

    # Calculate simularity between vector a & vector b.
    def simularity(self, a, b):
        return np.dot(a, b)/np.sqrt(np.dot(a, a) * np.dot(b, b))

    # Recommender using item-based method.
    def ItemBasedRecommend(self):

        # For each movie, predict users' rate.
        for movieIdx in range(self.movieCount):

            # Record Mean Value
            gradeArray = self.originalMatrix[movieIdx]
            mean = np.mean(gradeArray[gradeArray > 0])

            # Calculate simularity.
            print("Movie Idx: ", movieIdx)
            print(self.normalizedMatrix[movieIdx])
            simVec = np.zeros(self.movieCount, dtype=np.float32)
            for movieIdx2 in range(self.movieCount):
                simVec[movieIdx2] = self.simularity(
                    self.normalizedMatrix[movieIdx], self.normalizedMatrix[movieIdx2])
            
            # np.argsort simularities.
            simOrder = np.argsort(-simVec)              # Sort from large to small.
            simOrder = simOrder[:np.sum(simVec >= 0)]   # Only consider sim >= 0.
            print(simVec, simOrder)
            
            for userIdx in range(self.userCount):

                # For each user who hasn't graded the movie.
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
                            predGrade += simVec[neighbor] * self.originalMatrix[neighbor][userIdx]
                        predGrade /= simSum
                    else:
                        predGrade = mean
                    predGrade = round(predGrade)

                    self.originalMatrix[movieIdx][userIdx] = predGrade
                    print("(Movie, User): ({0}, {1}). Result: {2}".format(movieIdx, userIdx, predGrade))

    def debugOutput(self):
        print("Recommend Matrix:\n", self.originalMatrix)

    def saveRecommendMatrix(self):
        np.save(self.dataDir + 'Item_based_recommend_matrix.npy', self.originalMatrix)

    def run(self):
        self.readData()
        self.ItemBasedRecommend()
        self.saveRecommendMatrix()

if __name__ == '__main__':
    recommender = ItemBased(2)
    recommender.run()
