import numpy as np
import numpy.typing as npt
import fractions as fr

from collections import deque
from matplotlib import pyplot as plt 


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
        
        return res >= 0

    def _det(self, other):
        return _det3x3(np.stack([self.vec3(), other.vec3(), np.array([0, 0, 1])]))

    def __gt__(self, other) -> bool:
        mdet = self._det(other)
        if mdet > 0:
            return True
        elif mdet == 0 and self.vec3().dot(self.vec3()) > other.vec3().dot(other.vec3()):
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
        mhull = self.chull()
        center = self._get_center()

        min_val = float("inf")
        min_idx = -1
        for i in range(len(mhull)):
            x1 = mhull[i-1].v
            x2 = mhull[i].v
            norm = self._get_line_param(x1, x2)[0] + fr.Fraction()
            val = norm.dot(center - x1)**2 / norm.dot(norm)
            if min_val > val:
                min_val = val
                min_idx = i

        return mhull[min_idx-1].idx, mhull[min_idx].idx 


    def chull(self):
        q = self._get_iner_point()
        if q in self.vlist:
            self.vlist.remove(q)
        
        _SentralVertex.center = q.v
        self.vlist.sort()
        _SentralVertex.center = np.zeros((2))
        
        stack = deque()
        stack.append(q)
        stack.append(self.vlist[0])
        for v3 in self.vlist[1:]:
            v2 = stack[-1]
            v1 = stack[-2]
            while not _SentralVertex.is_right(v1, v2, v3):
                stack.pop()
                v2 = stack[-1]
                v1 = stack[-2]
            stack.append(v3)
        
        self.vlist.append(q)
        return list(stack)

    def _get_line_param(self, x1: npt.NDArray, x2: npt.NDArray):
        
        norm = x1 - x2
        norm[1] *= -1
        
        return norm, -norm.dot(x1)

    def _get_iner_point(self):
        
        min_q = self.vlist[0]
        for v in self.vlist[1:]:
            if min_q.v[1] > v.v[1] or (min_q.v[1] == v.v[1] and min_q.v[0] > v.v[0]):
                min_q = v
        
        return min_q

    def _get_center(self):
        
        ncenter = np.sum([v.v for v in self.vlist], axis=0) + fr.Fraction()
        center = ncenter / len(self.vlist)
        
        return center
        
    def rplot(self):

        q = self._get_iner_point()
        if q in self.vlist:
            self.vlist.remove(q)
        
        _SentralVertex.center = q.v
        self.vlist.sort()
        _SentralVertex.center = np.zeros((2))
        self.vlist.append(q)

        plt.figure()
        
        allcord = np.array([sv.v for sv in self.vlist])
        ann = [str(i) for i, sv in enumerate(self.vlist)]
        plt.scatter(allcord[:, 0], allcord[:, 1])
        for i, x in enumerate(allcord):
             plt.annotate(str(i), x)
        
        plt.scatter([q.v[0]], [q.v[1]])
        
        mhull = self.chull()
        hcord = np.array([v.v for v in mhull])
        plt.fill(hcord[:, 0], hcord[:, 1], fill=False)
        
        cen = self._get_center()
        plt.scatter([cen[0]], [cen[1]])
        print(self.min_splane())

        plt.show()