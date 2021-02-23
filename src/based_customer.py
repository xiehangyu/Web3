import numpy as np
def normalization(array_score):
    temp_array = array_score[array_score>0]
    mean = temp_array.mean()
    print('the mean is {0}'.format(mean))
    result = map(lambda x: mean if x ==0 else x,  array_score)
    return np.array(list(result))-mean 

def similarity(a, b):
    a = normalization(a)
    b = normalization(b)
    return np.dot(a, b)/np.sqrt(np.dot(a,a)*np.dot(b,b))