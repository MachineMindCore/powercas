import math
import cmath

# built-in
builtin_print = print
builtin_map = map
builtin_sum = sum
builtin_abs = abs
builtin_round = round


EULER = 2.7182818284
PI = 3.1415926535
NUMBERS = (int, float, complex)

### EXTREME WARNING:
###     Array is not squeezable selected

def array(data, dtype=float, decimals_repr=5):
    if isinstance(data, NUMBERS):
        return data
    if isinstance(data, list):
        ndim = Array(data, dtype).ndim
        if ndim == 2:
            return Array2D(data, dtype, decimals_repr)
        if ndim == 1:
            return Array1D(data, dtype, decimals_repr)
    raise ValueError("Non implemented dimension")

def matmul(arr_A, arr_B, dtype=float, decimals=5): 
    c = []
    for i in range(0,arr_A.shape[0]):
        temp=[]
        for j in range(0,arr_B.shape[1]):
            s = 0
            for k in range(0,arr_A.shape[1]):
                s += arr_A[i,k]*arr_B[k,j]
            temp.append(s)
        c.append(temp)
    return array(c, dtype, decimals)

def zeros(shape, dtype=float, decimals=5):
    ndim = len(shape)
    if ndim == 2:
        data = [[0 for _ in range(shape[1])] for i in range(shape[0])]
    elif ndim == 1:
        data = [0 for _ in range(shape[0])]
    else:
        raise ValueError("Not implemented dimension")
    return array(data, dtype, decimals)
    
def ones(shape, dtype=float, decimals=5):
    ndim = len(shape)
    if ndim == 2:
        data = [[1 for _ in range(shape[1])] for _ in range(shape[0])]
    elif ndim == 1:
        data = [1 for _ in range(shape[0])]
    else:
        raise ValueError("Not implemented dimension")
    return array(data, dtype, decimals)

def conjugate(arr):
    def simple_conjugate(number):
        return number.conjugate()
    
    if isinstance(arr, Array):
        new_arr = arr.copy()
        new_arr.dtype = complex
        new_arr._recursive_apply(simple_conjugate)
        return new_arr
    else:
        return arr.conjugate()

def round(arr, decimals=0):
    def simple_round(number, decimals):
        return builtin_round(number.real, decimals) + round(number.imag, decimals) * 1j
    
    if isinstance(arr, Array):
        new_array = arr.copy()
        new_array._recursive_apply(simple_round, decimals)
        return new_array.data
    elif isinstance(arr, complex):
        return builtin_round(arr.real, decimals) + round(arr.imag, decimals) * 1j
    else:
        return builtin_round(arr, decimals)

def abs(arr):
    if isinstance(arr, NUMBERS):
        return builtin_abs(arr)
    else:
        return apply(arr, builtin_abs)

def angle(arr):
    if isinstance(arr, NUMBERS):
        return cmath.phase(arr)
    else:
        return apply(arr, cmath.phase)
        

def csin(arr):
    if isinstance(arr, complex):
        return cmath.sin(arr)
    else:
        return apply(arr, cmath.sin)
    
def ccos(arr):
    if isinstance(arr, complex):
        return cmath.cos(arr)
    else:
        return apply(arr, cmath.cos)


def sin(arr):
    if isinstance(arr, (int, float)):
        return math.sin(arr)
    else:
        return apply(arr, math.sin)
    
def cos(arr):
    if isinstance(arr, (int, float)):
        return math.cos(arr)
    else:
        return apply(arr, math.cos)


def dot(arr_x, arr_y):
    result = zeros((arr_x.shape[0], arr_y.shape[1]))

    # Perform matrix multiplication
    for i in range(arr_x.shape[0]):
        for j in range(arr_y.shape[1]):
            for k in range(arr_y.shape[0]):
                result[i, j] += arr_x[i, k] * arr_y[k, j]
    return result

# only 2D
def append(arr_A, arr_B, axis=0, dtype=None):
    if dtype == None:
        dtype = arr_A.dtype
    dim_Ax = arr_A.shape[1]
    dim_Ay = arr_A.shape[0]
    dim_Bx = arr_B.shape[1]
    dim_By = arr_B.shape[0]
    if axis == 0:
        arr_new = zeros((dim_Ay + dim_By, dim_Ax), dtype=dtype)

        for i in range(dim_Ay):
            for j in range(dim_Ax):
                arr_new[i,j] = arr_A.data[i][j]

        for i in range(dim_By):
            for j in range(dim_Bx):
                arr_new[i+dim_Ay, j] = arr_B.data[i][j]

    elif axis == 1:
        arr_new = zeros((dim_Ay, dim_Ax + dim_Bx), dtype=dtype)

        for i in range(dim_Ay):
            for j in range(dim_Ax):
                arr_new[i,j] = arr_A.data[i][j]

        for i in range(dim_By):
            for j in range(dim_Bx):
                arr_new[i, j+dim_Ax] = arr_B.data[i][j]
    else:
        raise ValueError("Axis out of bounds")
    return arr_new

def apply(arr, func):
    arr_new = arr.copy()
    arr_new._recursive_apply(func)
    return arr_new

def exp(arr_A):
    return apply(arr_A, lambda x: EULER ** x)

def degrees(arr_A):
    return apply(arr_A, math.degrees)

def max(arr):
    m = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if m < arr.data[i][j]:
                m = arr.data[i][j]
    return m

# only vector
def sum(arr, index=0):
    value = 0
    if index==0:
        for i in range(arr.shape[0]):
            value += arr[i,0]
    if index==1:
        for j in range(arr.shape[1]):
            value += arr[0,j]
    return value
##########################################################################################################
def eliminate(r1, r2, col, target=0):
    fac = (r2[col]-target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]

def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i+1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise ValueError("Matrix is not invertible")
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a

def inverse(a):
    tmp = [[] for _ in a]
    for i,row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i])//2:])
    return ret

def inv(arr):
    inv_data = inverse(arr.data)
    try:
        return array(inv_data, arr.dtype)
    except:
        return array(inv_data, complex)

##########################################################################################################
class Array:
    def __init__(self, data, dtype=float, decimals_repr=3) -> None:
        
        if not isinstance(data, list):
            raise ValueError('Input data must be a list')
        
        self.data = data
        self.dtype = dtype
        self.decimals_repr = decimals_repr
        self.shape = self._compute_shape()
        self.ndim = len(self.shape)
        self._recursive_apply(dtype)

    def __repr__(self) -> str:
        repr_arr = self.copy()
        repr_arr._recursive_apply(round, repr_arr.decimals_repr)
        capture = ""
        if self.ndim == 1:
            capture = str(self.data)
        if self.ndim == 2:
            for i, line in enumerate(self.data):
                if i != 0:
                    capture += " "
                capture += str(line)
                if i != len(self.data)-1:
                    capture += "\n"
            return "[" + capture + "]" 

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        arr_new = self.copy()
        arr_new.dtype = self._check_operand_type(other)
        if isinstance(other, (int, float, complex)):
            arr_new._recursive_apply(lambda x: x + other)
        elif self.shape == other.shape:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    arr_new[i,j] = self.data[i][j] + other.data[i][j]
        else:
            raise ValueError("Must be same shape or scalar")
        return arr_new
    
    def __sub__(self, other):
        arr_new = self.copy()
        arr_new.dtype = self._check_operand_type(other)
        if isinstance(other, (int, float, complex)):
            arr_new._recursive_apply(lambda x: x - other)
        elif self.shape == other.shape:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    arr_new[i,j] = self.data[i][j] - other.data[i][j]
        else:
            raise ValueError("Must be same shape or scalar")
        return arr_new
    

    def __mul__(self, other):
        arr_new = self.copy()
        arr_new.dtype = self._check_operand_type(other)
        if isinstance(other, (int, float, complex)):
            arr_new._recursive_apply(lambda x: x * other)
        elif self.shape == other.shape:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    arr_new[i,j] = arr_new.data[i][j] * other.data[i][j]
        else:
            raise ValueError("Must be same shape or scalar")
        return arr_new

    def __truediv__(self, other):
        arr_new = self.copy()
        arr_new.dtype = self._check_operand_type(other)
        if isinstance(other, (int, float, complex)):
            arr_new._recursive_apply(lambda x: x * other)
        elif self.shape == other.shape:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    arr_new[i,j] = self.data[i][j] / other.data[i][j]
        else:
            raise ValueError("Must be same shape or scalar")
        return arr_new

    def _compute_shape(self):
        arr = self.data
        shape = []
        while isinstance(arr, list):
            shape.append(len(arr))
            arr = arr[0] if arr else None
        return tuple(shape)

    def _recursive_apply(self, func, *args, **kwargs) -> None:
        def _apply(data, func, *args, **kwargs):
            if isinstance(data, list):
                return [_apply(element, func, *args, **kwargs) for element in data]
            else:
                return func(data, *args, **kwargs)
                
        self.data = _apply(self.data, func, *args, **kwargs)
        return

    def _check_operand_type(self, other):
        other_type = type(other) if not isinstance(other, Array) else other.dtype
        if other_type == complex or self.dtype == complex:
            return complex
        return self.dtype

    def copy(self):
        dtype = type(self.data[0][0])
        new = array(data=self.data, dtype=dtype, decimals_repr=self.decimals_repr)
        return new
    
    def from_index(self, idx_list):
        result = [self.data[i] for i in idx_list]
        return array(result, dtype=self.dtype, decimals_repr=self.decimals_repr)
    
    def to_index(self, idx_list, values):
        arr_new = self.copy()
        for k, i in enumerate(idx_list):
            arr_new.data[i] = values.data[k]
        return arr_new

class Array2D(Array):
    def __init__(self, data, dtype=float, decimals_repr=3) -> None:
        super().__init__(data, dtype, decimals_repr)

    def __getitem__(self, indices):
        x, y = indices
        return array(self.data[x][y], dtype=self.dtype, decimals_repr=self.decimals_repr)
    
    def __setitem__(self, indices, value):
        x, y = indices
        self.data[x][y] = value
        return

    def transpose(self):
        result = zeros((self.shape[1], self.shape[0]), dtype=self.dtype, decimals=self.decimals_repr)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[j,i] = self.data[i][j]
        return result
    

class Array1D(Array):
    def __repr__(self) -> str:
        return str(self.data)

    def __getitem__(self, indices):
        return array(self.data[indices], dtype=self.dtype, decimals_repr=self.decimals_repr)
    
    def __setitem__(self, indices, value):
        self.data[indices] = value
        return
    
    def copy(self):
        return array(self.data, self.dtype, self.decimals_repr)


class Scalar:
    def __init__(self, value) -> None:
        self.value = value
    
    def __float__(self):
        if isinstance(self.value, float):
            return Complex(self.value + 0j)
        elif isinstance(self.value, complex):
            return Float(self.value)

class Complex(Scalar):
    def __init__(self, value) -> None:
        super().__init__(complex(value))
    

class Float(Scalar):
    def __init__(self, value) -> None:
        super().__init__(float(value))