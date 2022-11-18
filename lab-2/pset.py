import numpy as np
import numpy.typing as npt
import typing as tp
import fractions as fr

from collections import deque
from matplotlib import pyplot as plt 

def _det2x2(matrix: npt.NDArray):
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

def _det3x3(matrix: npt.NDArray):
    x = (matrix[0, 0] * matrix[1, 1] * matrix[2, 2]) + (matrix[1, 0] * matrix[2, 1] * matrix[0, 2]) + (matrix[2, 0] * matrix[0, 1] * matrix[1, 2])
    y = (matrix[0, 2] * matrix[1, 1] * matrix[2, 0]) + (matrix[1, 2] * matrix[2, 1] * matrix[0, 0]) + (matrix[2, 2] * matrix[0, 1] * matrix[1, 0])
    
    return x - y

class _SentralVertex:

    center: npt.NDArray = np.zeros((2))

    def __init__(self, vertex: npt.NDArray, idx: int = None) -> None:
        
        self.v = vertex
        self.idx = idx

    def vec3(self):
        vec2 = self.v - _SentralVertex.center
        return np.array([vec2[0], vec2[1], 0])

    def is_right(a, b, c):
        tmp_cen = _SentralVertex.center
        _SentralVertex.center = b.v
        res = a._det(c)
        _SentralVertex.center = tmp_cen
        
        return res

    def _det(self, other):
        return _det3x3(np.stack([self.vec3(), other.vec3(), np.array([0, 0, 1])]))

    def __gt__(self, other) -> bool:
        if self._det(other) > 0:
            return True
        elif self.vec3().dot(self.vec3()) > other.vec3().dot(other.vec3()):
            return True
        return False

    def __ge__(self, other) -> bool:
        return self.__gt__(other) or all(self.v == other.v)
    
    def __le__(self, other) -> bool:
        return not self.__gt__(other)
    
    def __lt__(self, other) -> bool:
        return not self.__ge__(other)

    def __eq__(self, __o: object) -> bool:
        return all(self.v == (__o.v if isinstance(__o, _SentralVertex) else __o))

    def __str__(self) -> str:
        return f"({self.idx}, {self.v})"
    
    def __repr__(self) -> str:
        return str(self)
    

class PointSet:

    def __init__(self, vlist: npt.NDArray) -> None:
        self.vlist = [_SentralVertex(v, i) for i, v in enumerate(vlist)]

    def min_splane(self):
        pass

    def chull(self):
        q = self._get_iner_point()
        if q in self.vlist:
            self.vlist.remove(q)
        
        _SentralVertex.center = q.v
        self.vlist.sort()
        _SentralVertex.center = np.zeros((2))
        
        stack = deque()
        stack.append(self.vlist[0])
        stack.append(self.vlist[1])
        for v3 in self.vlist[2:]:
            v2 = stack[-2]
            v1 = stack[-1]
            while _SentralVertex.is_right(v1, v2, v3):
                stack.pop()
                v2 = stack[-2]
                v1 = stack[-1]
            stack.append(v3)
        
        return stack

    def _get_iner_point(self):
        
        v1, v2 = self.vlist[:2]
        for i in range(2, len(self.vlist)):
            v3 = self.vlist[i]
            if _det2x2(np.stack([v1.v - v3.v, v2.v - v3.v])) != 0:
                break
        
        v1, v2, v3 = [self.vlist[k] for k in [1, 2, i]]

        return _SentralVertex((v1.v + v2.v + v3.v + fr.Fraction()) / 3)

    def rplot(self):
        plt.figure()
        
        allcord = np.array([sv.v for sv in self.vlist])
        marks = [str(i) for i, sv in enumerate(self.vlist)]
        plt.scatter(allcord[:, 0], allcord[:, 1], marker=marks)
        
        mhull = self.chull()
        cord = np.array([sv.v for sv in mhull])
        plt.scatter(cord[:,0], cord[:, 1], marker="h")
        plt.show()