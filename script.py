# imports
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit

# initialize seed
np.random.seed(0)

@jit
def ExtractFeatures(images, cell_size, grid_size):
    '''
    split image into grid of smaller images and return those
    each feature is flattened into an array
    '''
    
    for i in [0, 1]: assert cell_size[i] * grid_size[i] == images[0].shape[i]
    
    feature_size = np.prod(cell_size)
    split_images = np.empty(shape=(0, feature_size))
    num_images, height, width = images.shape
    
    for i in range(0, height, cell_size[0]):
        for j in range(0, width, cell_size[1]):
            split_images = np.concatenate((split_images, images[:, i:i+cell_size[0], j:j+cell_size[1]].reshape((-1, feature_size))))
    
    return split_images

@jit
def KMeansClustering(features, k, eps):
    '''
    return cluster centers of features by applying K-Means clustering and the corresponding BOW features
    '''
    
    num_features, feature_length = features.shape
    
    change = float('inf')
    prev_k_means = np.array([features[np.random.randint(0, num_features-1)] for _ in range(k)])
    new_k_means = None
        
    num_iters = 0
        
    with tqdm() as pbar:
        while (change > eps):

            closest_mean = GetClosestMean(features, prev_k_means, k)

            new_k_means = np.zeros((k, feature_length))
            freq_cluster = np.zeros(k)

            for i in range(num_features):
                new_k_means[closest_mean[i]] += features[i]
                freq_cluster[closest_mean[i]] += 1

            new_k_means = np.array([new_k_means[i] / freq_cluster[i] if freq_cluster[i] > 0 else prev_k_means[i] for i in range(k)])
            change = np.mean(np.linalg.norm(new_k_means - prev_k_means, axis=1))
            prev_k_means = new_k_means
            
            num_iters += 1
            pbar.set_description(f'At iteration {num_iters}, Error: {change}')
            pbar.update()
                                          
    return new_k_means

@jit
def GetClosestMean(features, words, k, softness=1):
    '''
    gives indices of closest means (as per softness) to each of the features
    By default, no softness is considered
    '''
    
    distance = np.empty((features.shape[0], 0))
    for word in words:
        distance = np.concatenate((distance, np.linalg.norm(features - word, axis = 1, keepdims=True)), axis=1)

    closest_mean = np.argpartition(distance, softness, axis=1)[:, :softness]
    return closest_mean

@jit
def GetClosestFeature(features, means, k):
    '''
    gives index of closest feature to each of the means
    '''
    return np.array([np.argmin(np.linalg.norm(mean - features, axis=1)) for mean in means])

@jit
def ComputeHistogram(num_images, features, words, k, softness, softness_weights):
    '''
    generates the BOW features according to the words given
    k = len(words)
    '''
    
    closest_mean = GetClosestMean(features, words, k, softness)
    weights = np.zeros((num_images, k))
    
    for i, c in enumerate(closest_mean):
        weights[i % num_images][c] += softness_weights
    
    return weights

@jit
def MatchHistogram(features_train, train_labels, features_test, softness, softness_weights):
    '''
    matches histogram using soft assignment
    '''
    
    assert softness == len(softness_weights), "Inconsistent Softness and length of Softness weights List"
    
    def get_weighted_mode(arr):
        from collections import defaultdict
        score = defaultdict(lambda: 0)
        for i, val in enumerate(softness_weights): score[arr[i]] += val
        return max(score, key= lambda x: score[x])
    
    return train_labels[
            [get_weighted_mode(
                np.argpartition(np.linalg.norm(features_train - i, axis=1), softness)[:softness]
            )
        for i in tqdm(features_test)]
    ]

@jit
def CreateVisualDictionary(train_images, k, cell_size, grid_size, eps):
    '''
    return visual dictionary
    1) extract features
    2) make clusters on the space containing these smaller images
    3) treat each of the cluster center as a word
    4) the set of word represents the visual dictionary
    '''
    
    features = ExtractFeatures(train_images, cell_size, grid_size)
    means = KMeansClustering(features, k, eps)
    words = features[GetClosestFeature(features, means, k)]
    features_train = ComputeHistogram(len(train_images), features, words, k)
    return words, features_train

def display_results(predicted_labels, test_labels):
    df = pd.DataFrame(predicted_labels, columns=['Predicted'])
    df['True'] = test_labels

    classes = np.unique(test_labels)

    accuracy = np.mean(df['Predicted'] == df['True'])

    df_pred = [df[df['Predicted'] == i] for i in classes]
    precision = [np.mean(df_pred[i]['Predicted'] == df_pred[i]['True']) for i in classes]

    df_true = [df[df['True'] == i] for i in classes]
    recall = [np.mean(df_true[i]['Predicted'] == df_true[i]['True']) for i in classes]

    labels = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
    ]

    print(f'Accuracy: {accuracy:.4f}')
    stats = pd.DataFrame({
        'Class': labels,
        'Precision': precision,
        'Recall': recall
    })
    print(stats)
    

def train(k
        ,eps
        ,path
        ,softness_feature_construction
        ,softness_weights_feature_construction
        ,cell_size
        ,grid_size
        ,train_images
        ,train_labels
        ,test_images):
    '''
    train the classifier:
    1) creates visual dictionary
    2) saves dictionary
    3) features corresponding to train and test set with the dictionary
    '''
    
    try:
        os.mkdir(f'./{path}')
    except:
        print('Path already exists')
        
    dictionary, features_train = CreateVisualDictionary(train_images, k, cell_size, grid_size, eps)
    np.save(f'./{path}/dictionary', dictionary)
    np.save(f'./{path}/features_train', features_train)
    
    features_test = ComputeHistogram(len(test_images), 
                                     ExtractFeatures(test_images), 
                                     dictionary, 
                                     k, 
                                     softness_feature_construction, 
                                     softness_weights_feature_construction)
    
    np.save(f'./{path}/features_test', features_test)


def predict(path
        ,softness_feature_matching
        ,softness_weights_feature_matching
        ,test_images):
    
    dictionary = np.load(f'./{path}/dictionary.npy')
    features_train = np.load(f'./{path}/features_train.npy')
    features_test = np.load(f'./{path}/features_test.npy')
    
    predicted_labels = MatchHistogram(features_train, 
                                      train_labels, 
                                      features_test, 
                                      softness_feature_matching, 
                                      softness_weights_feature_matching)
    
    np.save(f'./{path}/predicted_labels', predicted_labels)
    return predicted_labels


def main(*args, **kwargs):
    # loading dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    
    # train
    if (is_train:
        train(k
            ,eps
            ,path
            ,softness_feature_construction
            ,softness_weights_feature_construction
            ,cell_size
            ,grid_size
            ,train_images
            ,train_labels
            ,test_images
        )
        
    # test
    predicted_labels = predict(path
        ,softness_feature_matching
        ,softness_weights_feature_matching
        ,test_images)
    
    
    # display result
    display_results(predicted_labels, test_labels)
        
if __name__=='__main__':
    def parse_config(val, datatype):
        if (datatype == 'str'): return val
        if (datatype == 'int'): return int(val)
        if (datatype == 'list'): return list(map(float, val.split(',')))
        if (datatype == 'bool'): return val=='True'
        
    with open('CONFIG', 'r') as config:
        while (True):
            try:    
                var, datatype, val = config.readline().split('=')
                globals()[var] = parse_config(val, datatype)
            except:
                break
    
    main(globals())