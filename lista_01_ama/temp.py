Python 3.12.0 (main, Oct 10 2023, 10:37:17) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import matplotlib.pyplot as plt
A = np.array([[2,0],[4,6],[8,2]])
print("Dimensão de A:", A.shape)
Dimensão de A: (3, 2)
A[0].shape
(2,)
print(np.random.rand(5,3))
[[0.16511774 0.64431075 0.42965148]
 [0.92587839 0.05835982 0.81762567]
 [0.41278857 0.11904975 0.25767054]
 [0.30356267 0.39527099 0.89500642]
 [0.33489434 0.14972475 0.15014458]]
print(np.random.randn(2,2))
[[ 0.29433798  0.21600826]
 [-0.11075899  0.70727101]]
A.mean()
3.6666666666666665
A.sum()
22
A.prod()
0
pressao_dataset = np.genfromtxt('D:\Documentos\UFC\Machine Learning\Materials\NotebookTutorialNumpy/pressão.txt', delimiter=',', skip_header=1)
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 13-16: truncated \UXXXXXXXX escape
pressao_dataset = np.genfromtxt('D:\Documentos\UFC\Machine Learning\Materials\NotebookTutorialNumpy/pressão.txt', delimiter=',', skip_header=1)
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 13-16: truncated \UXXXXXXXX escape
pressao_dataset = np.genfromtxt("D:\Documentos\UFC\Machine Learning\Materials\NotebookTutorialNumpy/pressão.txt", delimiter=',', skip_header=1)
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 13-16: truncated \UXXXXXXXX escape
pressao_dataset = np.genfromtxt(D:\Documentos\UFC\Machine Learning\Materials\NotebookTutorialNumpy/pressão.txt, delimiter=',', skip_header=1)
SyntaxError: invalid syntax
s = StringIO(u"D:\Documentos\UFC\Machine Learning\Materials\NotebookTutorialNumpy/pressão.txt")
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 13-16: truncated \UXXXXXXXX escape
pressao_dataset = np.genfromtxt('D:\\Documentos\\UFC\\Machine Learning\\Materials\\NotebookTutorialNumpy/pressão.txt', delimiter=',', skip_header=1)
pressao_dataset
array([[ 39., 144.],
       [ 47., 220.],
       [ 45., 138.],
       [ 47., 145.],
       [ 65., 162.],
       [ 46., 142.],
       [ 67., 170.],
       [ 42., 124.],
       [ 67., 158.],
       [ 56., 154.],
       [ 64., 162.],
       [ 56., 150.],
       [ 59., 140.],
       [ 34., 110.],
       [ 42., 128.],
       [ 48., 130.],
       [ 45., 135.],
       [ 17., 114.],
       [ 20., 116.],
       [ 19., 124.],
       [ 36., 136.],
       [ 50., 142.],
       [ 39., 120.],
       [ 21., 120.],
       [ 44., 160.],
       [ 53., 158.],
       [ 63., 144.],
       [ 29., 130.],
       [ 25., 125.],
       [ 69., 175.]])
peixe_dataset = np.genfromtxt('D:\\Documentos\\UFC\\Machine Learning\\Materials\\NotebookTutorialNumpy/peixe.txt', delimiter=',')
peixe_dataset
array([[  14.,   25.,  620.],
       [  28.,   25., 1315.],
       [  41.,   25., 2120.],
       [  55.,   25., 2600.],
       [  69.,   25., 3110.],
       [  83.,   25., 3535.],
       [  97.,   25., 3935.],
       [ 111.,   25., 4465.],
       [ 125.,   25., 4530.],
       [ 139.,   25., 4570.],
       [ 153.,   25., 4600.],
       [  14.,   27.,  625.],
       [  28.,   27., 1215.],
       [  41.,   27., 2110.],
       [  55.,   27., 2805.],
       [  69.,   27., 3255.],
       [  83.,   27., 4015.],
       [  97.,   27., 4315.],
       [ 111.,   27., 4495.],
       [ 125.,   27., 4535.],
       [ 139.,   27., 4600.],
       [ 153.,   27., 4600.],
       [  14.,   29.,  590.],
       [  28.,   29., 1305.],
       [  41.,   29., 2140.],
       [  55.,   29., 2890.],
       [  69.,   29., 3920.],
       [  83.,   29., 3920.],
       [  97.,   29., 4515.],
       [ 111.,   29., 4520.],
       [ 125.,   29., 4525.],
       [ 139.,   29., 4565.],
       [ 153.,   29., 4566.],
       [  14.,   31.,  590.],
       [  28.,   31., 1205.],
       [  41.,   31., 1915.],
       [  55.,   31., 2140.],
       [  69.,   31., 2710.],
       [  83.,   31., 3020.],
       [  97.,   31., 3030.],
       [ 111.,   31., 3040.],
       [ 125.,   31., 3180.],
       [ 139.,   31., 3257.],
       [ 153.,   31., 3214.]])
A
array([[2, 0],
       [4, 6],
       [8, 2]])
A.size
6
A.size - 1
5
A.size(axis=0)
Traceback (most recent call last):
  File "<pyshell#22>", line 1, in <module>
    A.size(axis=0)
TypeError: 'int' object is not callable
A[0].size
2
A.shape
(3, 2)
A.shape - 1
Traceback (most recent call last):
  File "<pyshell#25>", line 1, in <module>
    A.shape - 1
TypeError: unsupported operand type(s) for -: 'tuple' and 'int'
A.shape(keepdims)
Traceback (most recent call last):
  File "<pyshell#26>", line 1, in <module>
    A.shape(keepdims)
NameError: name 'keepdims' is not defined
As = np.copy(A.shape)
As
array([3, 2])
As - 1
array([2, 1])
peixe_shape = np.copy(peixe_dataset.shape)
peixe_shape
array([44,  3])
peixe_shape - 1
array([43,  2])
peixe_dataset.size
132
peixe_dataset.size(axis=0)
Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    peixe_dataset.size(axis=0)
TypeError: 'int' object is not callable
np.size(peixe_dataset, axis=0)
44
np.size(peixe_dataset, axis=1)
3
np.size(peixe_dataset, axis=2)
Traceback (most recent call last):
  File "<pyshell#37>", line 1, in <module>
    np.size(peixe_dataset, axis=2)
  File "C:\Users\agar32\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\core\fromnumeric.py", line 3260, in size
    return a.shape[axis]
IndexError: tuple index out of range
peixe_dataset[0].size
3
peixe_dataset[1].size
3
peixe_dataset[0, 1].size
1
peixe_dataset[:, 1].size
44
peixe_dataset[:, 1].size - 1
43
def ols(X, y):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

1.T
SyntaxError: invalid decimal literal
X = [1, peixe_dataset[:,[0,1]]






     ]
X
[1, array([[ 14.,  25.],
       [ 28.,  25.],
       [ 41.,  25.],
       [ 55.,  25.],
       [ 69.,  25.],
       [ 83.,  25.],
       [ 97.,  25.],
       [111.,  25.],
       [125.,  25.],
       [139.,  25.],
       [153.,  25.],
       [ 14.,  27.],
       [ 28.,  27.],
       [ 41.,  27.],
       [ 55.,  27.],
       [ 69.,  27.],
       [ 83.,  27.],
       [ 97.,  27.],
       [111.,  27.],
       [125.,  27.],
       [139.,  27.],
       [153.,  27.],
       [ 14.,  29.],
       [ 28.,  29.],
       [ 41.,  29.],
       [ 55.,  29.],
       [ 69.,  29.],
       [ 83.,  29.],
       [ 97.,  29.],
       [111.,  29.],
       [125.,  29.],
       [139.,  29.],
       [153.,  29.],
       [ 14.,  31.],
       [ 28.,  31.],
       [ 41.,  31.],
       [ 55.,  31.],
       [ 69.,  31.],
       [ 83.,  31.],
       [ 97.,  31.],
       [111.,  31.],
       [125.,  31.],
       [139.,  31.],
       [153.,  31.]])]
y = peixe_dataset[:, [2]]
y
array([[ 620.],
       [1315.],
       [2120.],
       [2600.],
       [3110.],
       [3535.],
       [3935.],
       [4465.],
       [4530.],
       [4570.],
       [4600.],
       [ 625.],
       [1215.],
       [2110.],
       [2805.],
       [3255.],
       [4015.],
       [4315.],
       [4495.],
       [4535.],
       [4600.],
       [4600.],
       [ 590.],
       [1305.],
       [2140.],
       [2890.],
       [3920.],
       [3920.],
       [4515.],
       [4520.],
       [4525.],
       [4565.],
       [4566.],
       [ 590.],
       [1205.],
       [1915.],
       [2140.],
       [2710.],
       [3020.],
       [3030.],
       [3040.],
       [3180.],
       [3257.],
       [3214.]])
artificial1d = np.genfromtxt('D:\\Documentos\\UFC\\Machine Learning\\lista_01_ama/artificial1d.csv', delimiter=',')
artificial1d
array([[-1.        , -2.08201726],
       [-0.93103448, -1.32698023],
       [-0.86206897, -1.10559772],
       [-0.79310345, -0.87394576],
       [-0.72413793, -0.28502695],
       [-0.65517241, -0.43115252],
       [-0.5862069 , -0.79475402],
       [-0.51724138, -0.88606806],
       [-0.44827586, -0.89989978],
       [-0.37931034, -0.86184365],
       [-0.31034483, -0.88805183],
       [-0.24137931, -1.23595129],
       [-0.17241379, -0.71956827],
       [-0.10344828, -0.45202286],
       [-0.03448276,  0.09889951],
       [ 0.03448276,  0.34896973],
       [ 0.10344828,  0.09747797],
       [ 0.17241379,  0.70019809],
       [ 0.24137931,  1.31051213],
       [ 0.31034483,  1.00177576],
       [ 0.37931034,  1.00318231],
       [ 0.44827586,  1.14910129],
       [ 0.51724138,  1.59220607],
       [ 0.5862069 ,  0.60909009],
       [ 0.65517241,  0.59441623],
       [ 0.72413793,  0.70300732],
       [ 0.79310345,  0.82332241],
       [ 0.86206897,  1.10646439],
       [ 0.93103448,  1.42295695],
       [ 1.        ,  2.30983768]])
artificial1d[:, [1]]
array([[-2.08201726],
       [-1.32698023],
       [-1.10559772],
       [-0.87394576],
       [-0.28502695],
       [-0.43115252],
       [-0.79475402],
       [-0.88606806],
       [-0.89989978],
       [-0.86184365],
       [-0.88805183],
       [-1.23595129],
       [-0.71956827],
       [-0.45202286],
       [ 0.09889951],
       [ 0.34896973],
       [ 0.09747797],
       [ 0.70019809],
       [ 1.31051213],
       [ 1.00177576],
       [ 1.00318231],
       [ 1.14910129],
       [ 1.59220607],
       [ 0.60909009],
       [ 0.59441623],
       [ 0.70300732],
       [ 0.82332241],
       [ 1.10646439],
       [ 1.42295695],
       [ 2.30983768]])
y = artificial1d[:, [1]]
X = [1, artificial1d[:, [0]]]
X
[1, array([[-1.        ],
       [-0.93103448],
       [-0.86206897],
       [-0.79310345],
       [-0.72413793],
       [-0.65517241],
       [-0.5862069 ],
       [-0.51724138],
       [-0.44827586],
       [-0.37931034],
       [-0.31034483],
       [-0.24137931],
       [-0.17241379],
       [-0.10344828],
       [-0.03448276],
       [ 0.03448276],
       [ 0.10344828],
       [ 0.17241379],
       [ 0.24137931],
       [ 0.31034483],
       [ 0.37931034],
       [ 0.44827586],
       [ 0.51724138],
       [ 0.5862069 ],
       [ 0.65517241],
       [ 0.72413793],
       [ 0.79310345],
       [ 0.86206897],
       [ 0.93103448],
       [ 1.        ]])]
ols(X, y)
Traceback (most recent call last):
  File "<pyshell#63>", line 1, in <module>
    ols(X, y)
  File "<pyshell#44>", line 2, in ols
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
AttributeError: 'list' object has no attribute 'T'
X.T
Traceback (most recent call last):
  File "<pyshell#64>", line 1, in <module>
    X.T
AttributeError: 'list' object has no attribute 'T'
A.T
array([[2, 4, 8],
       [0, 6, 2]])
X = artificial1d[:, [0]]
ols(X, y)
array([[1.57486517]])
np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
array([[1.57486517]])
np.dot(X.T, X) @ np.dot(X.T, y)
array([[179.95783917]])
X.T @ X @ X.T @ y
array([[179.95783917]])
def ols(X, y):
    return np.linalg.solve(X.T@X, X.T@y)

ols(X, y)
array([[1.57486517]])
def wPred(X, y):
    return X.T@X @ X.T@y

wPred(X, y)
array([[179.95783917]])
X
array([[-1.        ],
       [-0.93103448],
       [-0.86206897],
       [-0.79310345],
       [-0.72413793],
       [-0.65517241],
       [-0.5862069 ],
       [-0.51724138],
       [-0.44827586],
       [-0.37931034],
       [-0.31034483],
       [-0.24137931],
       [-0.17241379],
       [-0.10344828],
       [-0.03448276],
       [ 0.03448276],
       [ 0.10344828],
       [ 0.17241379],
       [ 0.24137931],
       [ 0.31034483],
       [ 0.37931034],
       [ 0.44827586],
       [ 0.51724138],
       [ 0.5862069 ],
       [ 0.65517241],
       [ 0.72413793],
       [ 0.79310345],
       [ 0.86206897],
       [ 0.93103448],
       [ 1.        ]])
artificial1d
array([[-1.        , -2.08201726],
       [-0.93103448, -1.32698023],
       [-0.86206897, -1.10559772],
       [-0.79310345, -0.87394576],
       [-0.72413793, -0.28502695],
       [-0.65517241, -0.43115252],
       [-0.5862069 , -0.79475402],
       [-0.51724138, -0.88606806],
       [-0.44827586, -0.89989978],
       [-0.37931034, -0.86184365],
       [-0.31034483, -0.88805183],
       [-0.24137931, -1.23595129],
       [-0.17241379, -0.71956827],
       [-0.10344828, -0.45202286],
       [-0.03448276,  0.09889951],
       [ 0.03448276,  0.34896973],
       [ 0.10344828,  0.09747797],
       [ 0.17241379,  0.70019809],
       [ 0.24137931,  1.31051213],
       [ 0.31034483,  1.00177576],
       [ 0.37931034,  1.00318231],
       [ 0.44827586,  1.14910129],
       [ 0.51724138,  1.59220607],
       [ 0.5862069 ,  0.60909009],
       [ 0.65517241,  0.59441623],
       [ 0.72413793,  0.70300732],
       [ 0.79310345,  0.82332241],
       [ 0.86206897,  1.10646439],
       [ 0.93103448,  1.42295695],
       [ 1.        ,  2.30983768]])
[1, artificial1d[:, [0]]]
[1, array([[-1.        ],
       [-0.93103448],
       [-0.86206897],
       [-0.79310345],
       [-0.72413793],
       [-0.65517241],
       [-0.5862069 ],
       [-0.51724138],
       [-0.44827586],
       [-0.37931034],
       [-0.31034483],
       [-0.24137931],
       [-0.17241379],
       [-0.10344828],
       [-0.03448276],
       [ 0.03448276],
       [ 0.10344828],
       [ 0.17241379],
       [ 0.24137931],
       [ 0.31034483],
       [ 0.37931034],
       [ 0.44827586],
       [ 0.51724138],
       [ 0.5862069 ],
       [ 0.65517241],
       [ 0.72413793],
       [ 0.79310345],
       [ 0.86206897],
       [ 0.93103448],
       [ 1.        ]])]
[1, np.copy(artificial1d[:, [0]])]
[1, array([[-1.        ],
       [-0.93103448],
       [-0.86206897],
       [-0.79310345],
       [-0.72413793],
       [-0.65517241],
       [-0.5862069 ],
       [-0.51724138],
       [-0.44827586],
       [-0.37931034],
       [-0.31034483],
       [-0.24137931],
       [-0.17241379],
       [-0.10344828],
       [-0.03448276],
       [ 0.03448276],
       [ 0.10344828],
       [ 0.17241379],
       [ 0.24137931],
       [ 0.31034483],
       [ 0.37931034],
       [ 0.44827586],
       [ 0.51724138],
       [ 0.5862069 ],
       [ 0.65517241],
       [ 0.72413793],
       [ 0.79310345],
       [ 0.86206897],
       [ 0.93103448],
       [ 1.        ]])]
np.ones(artificial1d[:, [0]].size, 1)
Traceback (most recent call last):
  File "<pyshell#82>", line 1, in <module>
    np.ones(artificial1d[:, [0]].size, 1)
  File "C:\Users\agar32\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\core\numeric.py", line 191, in ones
    a = empty(shape, dtype, order)
TypeError: Cannot interpret '1' as a data type
artificial1d[:, [0]].size
30
temp = artificial1d[:, [0]].size
np.ones(temp, 1)
Traceback (most recent call last):
  File "<pyshell#85>", line 1, in <module>
    np.ones(temp, 1)
  File "C:\Users\agar32\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\core\numeric.py", line 191, in ones
    a = empty(shape, dtype, order)
TypeError: Cannot interpret '1' as a data type
tempV = np.ones(temp, 1)
Traceback (most recent call last):
  File "<pyshell#86>", line 1, in <module>
    tempV = np.ones(temp, 1)
  File "C:\Users\agar32\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\core\numeric.py", line 191, in ones
    a = empty(shape, dtype, order)
TypeError: Cannot interpret '1' as a data type
np.ones((temp, 1))
array([[1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.]])
X = np.append(np.ones((temp, 1)), artificial1d[:, [0]])
X
array([ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
       -1.        , -0.93103448, -0.86206897, -0.79310345, -0.72413793,
       -0.65517241, -0.5862069 , -0.51724138, -0.44827586, -0.37931034,
       -0.31034483, -0.24137931, -0.17241379, -0.10344828, -0.03448276,
        0.03448276,  0.10344828,  0.17241379,  0.24137931,  0.31034483,
        0.37931034,  0.44827586,  0.51724138,  0.5862069 ,  0.65517241,
        0.72413793,  0.79310345,  0.86206897,  0.93103448,  1.        ])
X = np.append(np.ones((temp, 1)), artificial1d[:, [0]], axis=1)
X
array([[ 1.        , -1.        ],
       [ 1.        , -0.93103448],
       [ 1.        , -0.86206897],
       [ 1.        , -0.79310345],
       [ 1.        , -0.72413793],
       [ 1.        , -0.65517241],
       [ 1.        , -0.5862069 ],
       [ 1.        , -0.51724138],
       [ 1.        , -0.44827586],
       [ 1.        , -0.37931034],
       [ 1.        , -0.31034483],
       [ 1.        , -0.24137931],
       [ 1.        , -0.17241379],
       [ 1.        , -0.10344828],
       [ 1.        , -0.03448276],
       [ 1.        ,  0.03448276],
       [ 1.        ,  0.10344828],
       [ 1.        ,  0.17241379],
       [ 1.        ,  0.24137931],
       [ 1.        ,  0.31034483],
       [ 1.        ,  0.37931034],
       [ 1.        ,  0.44827586],
       [ 1.        ,  0.51724138],
       [ 1.        ,  0.5862069 ],
       [ 1.        ,  0.65517241],
       [ 1.        ,  0.72413793],
       [ 1.        ,  0.79310345],
       [ 1.        ,  0.86206897],
       [ 1.        ,  0.93103448],
       [ 1.        ,  1.        ]])
>>> ols(X, y)
array([[0.06761792],
       [1.57486517]])
>>> y
array([[-2.08201726],
       [-1.32698023],
       [-1.10559772],
       [-0.87394576],
       [-0.28502695],
       [-0.43115252],
       [-0.79475402],
       [-0.88606806],
       [-0.89989978],
       [-0.86184365],
       [-0.88805183],
       [-1.23595129],
       [-0.71956827],
       [-0.45202286],
       [ 0.09889951],
       [ 0.34896973],
       [ 0.09747797],
       [ 0.70019809],
       [ 1.31051213],
       [ 1.00177576],
       [ 1.00318231],
       [ 1.14910129],
       [ 1.59220607],
       [ 0.60909009],
       [ 0.59441623],
       [ 0.70300732],
       [ 0.82332241],
       [ 1.10646439],
       [ 1.42295695],
       [ 2.30983768]])
>>> y = artificial1d[:, [1]]
>>> ols(X, y)
array([[0.06761792],
       [1.57486517]])
>>> def rmse(y, pred):
...     return np.sqrt(np.mean((y - pred) ** 2))
... 
>>> predFull = ols(X, y)
rmse(y, predFull[:, [1]])
Traceback (most recent call last):
  File "<pyshell#99>", line 1, in <module>
    rmse(y, predFull[:, [1]])
IndexError: index 1 is out of bounds for axis 1 with size 1
predFull[:, [1]]
Traceback (most recent call last):
  File "<pyshell#100>", line 1, in <module>
    predFull[:, [1]]
IndexError: index 1 is out of bounds for axis 1 with size 1
predFull[:, 1]
Traceback (most recent call last):
  File "<pyshell#101>", line 1, in <module>
    predFull[:, 1]
IndexError: index 1 is out of bounds for axis 1 with size 1
predFull
array([[0.06761792],
       [1.57486517]])
predFull[1]
array([1.57486517])
rmse(y, predFull[1])
1.8354685204301637
predFull = ols(X, y)
rmse(y, predFull[1])
1.8354685204301637
xRaw = artificial1d[:, [0]]
xµ = np.mean(xRaw, axis = 0)
xSigma = np.std(xRaw, axis = 0)
xNormalized = (xRaw - x) / X_std
KeyboardInterrupt
xNormalized = ((xRaw - xµ) / xSigma)
X = np.append(np.ones((artificial1d[:, [0]].shape[0], 1)), xNormalized, axis=1)
X
array([[ 1.        , -1.67524673],
       [ 1.        , -1.55971247],
       [ 1.        , -1.44417822],
       [ 1.        , -1.32864396],
       [ 1.        , -1.2131097 ],
       [ 1.        , -1.09757545],
       [ 1.        , -0.98204119],
       [ 1.        , -0.86650693],
       [ 1.        , -0.75097267],
       [ 1.        , -0.63543842],
       [ 1.        , -0.51990416],
       [ 1.        , -0.4043699 ],
       [ 1.        , -0.28883564],
       [ 1.        , -0.17330139],
       [ 1.        , -0.05776713],
       [ 1.        ,  0.05776713],
       [ 1.        ,  0.17330139],
       [ 1.        ,  0.28883564],
       [ 1.        ,  0.4043699 ],
       [ 1.        ,  0.51990416],
       [ 1.        ,  0.63543842],
       [ 1.        ,  0.75097267],
       [ 1.        ,  0.86650693],
       [ 1.        ,  0.98204119],
       [ 1.        ,  1.09757545],
       [ 1.        ,  1.2131097 ],
       [ 1.        ,  1.32864396],
       [ 1.        ,  1.44417822],
       [ 1.        ,  1.55971247],
       [ 1.        ,  1.67524673]])
yRaw = artificial1d[:, [1]]
yMean = np.mean(yRaw, axis = 0)
ySigma = np.std(yRaw, axis = 0)
y = ((yRaw - yMean)/ySigma)
w = np.linalg.inv(X.T@X) @ X.T @ y
w
array([[-1.38777878e-17],
       [ 8.97493907e-01]])
pred = X @ w
pred = pred * y_std + y_mean
Traceback (most recent call last):
  File "<pyshell#120>", line 1, in <module>
    pred = pred * y_std + y_mean
NameError: name 'y_std' is not defined
pred = pred * ySigma + yMean
pred
array([[-1.50724724],
       [-1.39863585],
       [-1.29002446],
       [-1.18141307],
       [-1.07280168],
       [-0.96419029],
       [-0.8555789 ],
       [-0.74696751],
       [-0.63835612],
       [-0.52974473],
       [-0.42113333],
       [-0.31252194],
       [-0.20391055],
       [-0.09529916],
       [ 0.01331223],
       [ 0.12192362],
       [ 0.23053501],
       [ 0.3391464 ],
       [ 0.44775779],
       [ 0.55636918],
       [ 0.66498057],
       [ 0.77359197],
       [ 0.88220336],
       [ 0.99081475],
       [ 1.09942614],
       [ 1.20803753],
       [ 1.31664892],
       [ 1.42526031],
       [ 1.5338717 ],
       [ 1.64248309]])
def mse(y, pred):
    return np.mean((y - pred) ** 2)

mse(y, pred)
0.20089040610178371
