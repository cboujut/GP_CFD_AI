# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:12:56 2021

@author: t.corbin
"""

def Toto_function(AoA,ca1,ca2,ce1,ce2):
    with open('X_values.txt', 'r') as Xs:
        X = Xs.readlines()
        X = X[AoA].split(' ')
    summ = float(X[0])*ca1 + float(X[1])*ca2 + float(X[2])*ce1 + float(X[3])*ce2
    error = abs(100 - summ)
    return error
