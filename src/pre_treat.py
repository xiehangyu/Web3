import numpy as np
from sklearn.preprocessing import LabelEncoder


class based_on_object:
    UserEncoder = LabelEncoder()
    MovieEncoder = LabelEncoder()
    Original_score_matrix = []
    Normalized_matrix = []
    Number_of_users = 0
    Number_of_movies = 0
    TrainingDataDir = '../data/training.dat'
    SavingDir = '../data/'
    
    MovieEncodingDict = {}
    UserEncodingDict = {}# this two dictionaries are to speed up the procedure. the dictionary is {Original Index:EncoderLabel}

    def __init__(self):
        pass        


    def initial_encoders(self):
        """ 
        This function will initial the UserEncoder, MovideEncoder, Number_of_users and Number_of_movies
        """
        file_read = open(self.TrainingDataDir, 'r', encoding='utf-8').read().split('\n')
        print("The total number of records is {0}".format(len(file_read)))
        temp_movie_ls = []
        temp_user_ls = []
        for i in range(len(file_read) - 1):
            if i % 10000 == 0:
                print("Reading records... Current index:", i)
            temp_movie_ls.append(int(file_read[i].split(',')[1]))
            temp_user_ls.append(int(file_read[i].split(',')[0]))
        self.Number_of_users = len(set(temp_user_ls))
        self.Number_of_movies = len(set(temp_movie_ls))
        self.UserEncoder.fit(temp_user_ls)
        self.MovieEncoder.fit(temp_movie_ls)

    def initialDict(self):
        """
        This function will initial the dictionaries
        """
        for i in range(self.Number_of_movies):
            if i % 10000 == 0:
                print("generating movie dict... Current index:", i)
            self.MovieEncodingDict[int(self.MovieEncoder.inverse_transform([i]))] = i
        for i in range(self.Number_of_users):
            if i % 1000 == 0:
                print("generating user dict... Current index:", i)
            self.UserEncodingDict[int(self.UserEncoder.inverse_transform([i]))] = i

    def initial_Original_score_matrix(self):
        """
        This function will initial the Original Score Matrix, the form is as the P46 in the 14th courseware.
        -1 represents there is no record for that movie and user
        """
        self.Original_score_matrix = -1*np.ones([self.Number_of_movies, self.Number_of_users],dtype='int8')
        file_read = open(self.TrainingDataDir, 'r', encoding="utf-8")
        for line in file_read.readlines():
            line = line.split(',')
            movie_ID = int(line[1])
            user_ID = int(line[0])
            score = int(line[2])
            self.Original_score_matrix[self.MovieEncodingDict[movie_ID], self.UserEncodingDict[user_ID]] = score 

    def save_Original_score_matrix(self):
        """
        This function will save the original score matrix
        """
        np.save(self.SavingDir+"Original_score_matrix.npy",self.Original_score_matrix)

    def load_Original_score_matrix(self):
        """
        This function will load the original score matrix
        """
        self.Original_score_matrix = np.load(self.SavingDir+"Original_score_matrix.npy", allow_pickle = True)

    def initial_Normalized_matrix(self):
        """
        This function will initial the normalize matrix of the movie vectors. This function
        requires the original score matrix is created
        If a movie doesn't have any records, that row will set to 0
        """
        self.Normalized_matrix = np.zeros([self.Number_of_movies, self.Number_of_users], dtype = 'float32')
        for i in range(len(self.Original_score_matrix)):
            if i % 1000 == 0:
                print("Normalizing... Current index: ", i)
            temp_array = self.Original_score_matrix[i]
            mean = temp_array[temp_array>=0].mean()
            result = np.array(list(map(lambda x: mean if x == -1 else x, temp_array)))
            for j in range(len(result)):
                self.Normalized_matrix[i][j] = result[j] - mean
    
    def save_Normalized_matrix(self):
        """
        This function will save the Normalized matrix
        """
        np.save(self.SavingDir+"Normalized_matrix.npy", self.Normalized_matrix)

    def load_Normalized_matrix(self):
        """
        This function will load the Normalized matrix
        """
        self.Normalized_matrix = np.load(self.SavingDir+"Normalized_matrix.npy", allow_pickle = True)

    def save_dict(self):
        """
        This function will save the dicts.
        """
        np.save(self.SavingDir+"user_encoding_dict.npy",
                self.UserEncodingDict)
        np.save(self.SavingDir+"movie_encoding_dict.npy",
                self.MovieEncodingDict)

    def load_dict(self):

        self.MovieEncodingDict = np.load(
            self.SavingDir + 'movie_encoding_dict.npy', allow_pickle=True).item()
        self.UserEncodingDict = np.load(
            self.SavingDir + 'user_encoding_dict.npy', allow_pickle=True).item()

    def General_Initialization(self):
        """
        This function is for the general initialization of the class
        """ 
        self.initial_encoders()
        self.initialDict()
        self.save_dict()
        self.initial_Original_score_matrix()
        self.save_Original_score_matrix()
        print('the original score matrix is now created and saved')
        self.initial_Normalized_matrix()
        self.save_Normalized_matrix()
        print('the normalized matrix is now created and saved')

    def Initialize_with_dict(self):
        self.initial_encoders()
        print('loading dict...')
        self.load_dict()
        self.initial_Original_score_matrix()
        self.save_Original_score_matrix()
        print('the original score matrix is now created and saved')
        self.initial_Normalized_matrix()
        self.save_Normalized_matrix()
        print('the normalized matrix is now created and saved')
def similarity (a, b):
    """
    Given two normalized vectors a and b, this function will return their similarity
    """
    return np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))

if __name__ == "__main__":
    preprocessor = based_on_object()
    preprocessor.Initialize_with_dict()
