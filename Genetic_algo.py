import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

def thomasCalculator(ca1,ca2,ce1,ce2, AoA):
    
    with open('X_values.txt', 'r') as Xs:
        X = Xs.readlines()
        X = X[AoA].split(' ')
        
    summ = float(X[0])*ca1 + float(X[1])*ca2 + float(X[2])*ce1 + float(X[3])*ce2
    
    error = abs(100 - summ)
    
    return error


def yashwantCalculator(model, ca1, ca2, ce1, ce2, AoA):
    
    #trained_model = tf.keras.models.load_model('dnn_model')
    inp = np.array([[AoA, ca1, ca2, ce1, ce2]])
    df = pd.DataFrame(inp)
    out = model.predict(df).flatten()
    
    return abs(out[0])

def liftCalculator(model, ca1, ca2, ce1, ce2, AoA):
    
    #trained_model = tf.keras.models.load_model('dnn_model')
    inp = np.array([[AoA, ca1, ca2, ce1, ce2]])
    df = pd.DataFrame(inp)
    out = model.predict(df).flatten()
    
    return out[0],out[1]

def parentGeneration(model, n_parents, AoA, method):
        
    #We check if we have an even amount of parents, if that's not the case we remove the last parent.
    
    if n_parents%2 != 0:
        n_parents -= 1
        
    list_parents = []
    
    for i in range(n_parents):
        
        ca1 = round(random.uniform(2, 20), 2)
        ca2 = round(random.uniform(0, 0.3), 2)
        ce1 = round(random.uniform(0.3, 10), 2)
        ce2 = round(random.uniform(5, 50), 1)
        
        if method == 'Thomas':
            list_parents.append([ca1, ca2, ce1, ce2, thomasCalculator(ca1, ca2, ce1, ce2, AoA)])
        elif method == 'Yashwant':
            list_parents.append([ca1, ca2, ce1, ce2, yashwantCalculator(model, ca1, ca2, ce1, ce2, AoA)])
        
    return list_parents


def newGeneration(model, parents, AoA, method) :
    
    n_parents = len(parents)
    n_iter = n_parents/2
    
    child = []
    newGen = []
    
    for i in range(int(n_iter)):
        
        random_dad = parents[np.random.randint(low = 0, high = n_parents - 1)]
        random_mom = parents[np.random.randint(low = 0, high = n_parents - 1)]
        
        dad_mask = np.random.randint(0, 2, size = np.array(random_dad).shape)
        mom_mask = np.logical_not(dad_mask)
        
        child = np.add(np.multiply(random_dad, dad_mask), np.multiply(random_mom, mom_mask))
        
        if method == 'Thomas':
            child[4] = thomasCalculator(child[0], child[1], child[2], child[3], AoA)
        elif method == 'Yashwant':
            child[4] = yashwantCalculator(model, child[0], child[1], child[2], child[3], AoA)
        
        newGen.append(child.tolist())
       
        if random_dad[4] >= random_mom[4]:
            newGen.append(random_dad)
        else:
            newGen.append(random_mom)
        
    return newGen


def mutatedGeneration(parents, model, AoA, method):
    
    size = len(parents) - 1
    mutated_gen = parents
    n_mutation = len(parents)/10
        
    for i in range(int(n_mutation)):
        
        rand1 = np.random.randint(0, size)
        rand2 = np.random.randint(0, 4)
            
        if rand2 == 0:
            mutated_gen[rand1][rand2] = round(random.uniform(2, 20), 2)
        elif rand2 == 1:
            mutated_gen[rand1][rand2] = round(random.uniform(0, 0.3), 2)
        elif rand2 == 2:
            mutated_gen[rand1][rand2] = round(random.uniform(0.3, 10), 2)
        elif rand2 == 3:
            mutated_gen[rand1][rand2] = round(random.uniform(5, 50), 1)
        
    return mutated_gen


def gen_algorithm(model, AoA, n_iteration, n_population, n_mutation, method):
    
    parent_gen = parentGeneration(model, n_population, AoA, method)
    best_child = parent_gen[0]
    list_best_child = []
    it = 0
    k = 0
    error_best = 500
    best_parent = 0
    
    for i in range(int(n_iteration)):
        parent_gen = newGeneration(model, parent_gen, AoA, method)
        k += 1
        
        if int(it) < int(n_mutation):
            rand = np.random.randint(0, 2)
            if rand == 1:
                parent_gen = mutatedGeneration(parent_gen, model, AoA, method)
                it+=1
                print('mutated')
                
        list_best_child.append(best_child[4])
        print('Iteration {}'.format(i))
        
        for j in parent_gen:
            if (j[4] < best_child[4]):
                best_child = j
                k = 0
        
        if k >= 3:
            for l in range(n_population):
                if (parent_gen[l][4] < error_best):
                    error_best = parent_gen[l][4]
                    best_parent = l
                    print(error_best)
                    
            print(parent_gen[best_parent])
            rand = np.random.randint(0,4)
            if rand == 0:
                parent_gen[best_parent][0] += 0.2
            elif rand == 1:
                parent_gen[best_parent][1] += 0.02
            elif rand == 2:
                parent_gen[best_parent][2] += 0.2
            elif rand == 3:
                parent_gen[best_parent][3] += 2
                
            if method == 'Thomas':
                parent_gen[best_parent][4] = thomasCalculator(parent_gen[best_parent][0], parent_gen[best_parent][1], parent_gen[best_parent][2], parent_gen[best_parent][3], AoA)
            elif method == 'Yashwant':
                parent_gen[best_parent][4] = yashwantCalculator(model, parent_gen[best_parent][0], parent_gen[best_parent][1], parent_gen[best_parent][2], parent_gen[best_parent][3], AoA)
                
            print(parent_gen[best_parent])
            k = 0
                    
                
    return (best_child, list_best_child)
    

trained_model = tf.keras.models.load_model('dnn_model')
cdModel = tf.keras.models.load_model('Cl_Cd_Model')
n_iter = input('Number of iteration: ')
population = input('How much population: ')
AoA = input('AoA: ')
n_mutation = input('Number of mutation: ')
method = input('Which method to use: ')

(answer, errors) = gen_algorithm(trained_model, int(AoA), int(n_iter), int(population), n_mutation, method)

ca1 = answer[0]
ca2 = answer[1]
ce1 = answer[2]
ce2 = answer[3]

cd,cl = liftCalculator(cdModel, ca1, ca2, ce1, ce2, int(AoA))

plt.plot(np.linspace(0, int(n_iter) - 1, int(n_iter)), errors)