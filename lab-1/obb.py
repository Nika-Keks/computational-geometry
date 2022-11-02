"""
algoritm:
    input: 
        - polygon [p1, p2, ..., pn]
        - criterion lambda (a, b) -> flaot | int
    
    output:   
        - [i1, i2, i3, i4] - vertex indexes

    description:
        - 
"""

import numpy as np
import typing as tp
import fractions as fr

from numpy import typing as npt

from  . import utils as mutils



__all__ = ["Poligon"]

def _perimeter(obb_v: npt.NDArray):
        
    wd = obb_v[0] - obb_v[1]
    hd = obb_v[1] - obb_v[2]
    hd = np.array([hd[1], -hd[0]])
    hd = hd * int(np.sign(wd.dot(hd)))

    sum_len = hd + wd

    return sum_len.dot(sum_len)
    
def _square(obb_v: npt.NDArray):

    wd = obb_v[0] - obb_v[1]
    hd = obb_v[1] - obb_v[2]

    return wd.dot(wd) * hd.dot(hd)



class Poligon:

    def __init__(self, vertex_list: np.ndarray, criterion = _perimeter, verbose=False) -> None:
        
        self.v_list = vertex_list.astype(np.int64)
        self.criterion = criterion
        self.verbose = verbose


    def get_obb(self):
        
        start_idxs = self._get_start_idxs()
        min_idxs = np.copy(start_idxs)
        min_cval = self._get_criterion(min_idxs)
        cur_idxs = self._next_idxs(start_idxs)
        
        if self.verbose:
                mutils.plot_obb(self.v_list, min_idxs, f"c = {min_cval}, ids={min_idxs}")

        while cur_idxs[0] != start_idxs[0]:
            cur_cval = self._get_criterion(cur_idxs)

            if self.verbose:
                mutils.plot_obb(self.v_list, cur_idxs, f"c = {cur_cval}, ids={cur_idxs}")

            if cur_cval < min_cval:
                min_cval = cur_cval
                min_idxs = np.copy(cur_idxs)

            cur_idxs = self._next_idxs(cur_idxs)

        return min_idxs

    def _get_start_idxs(self):

        idxs = np.zeros((4,), dtype=np.int32)
        idxs[0] = 1
        way = self.v_list[1] - self.v_list[0]
        norm_way = np.array([way[1], -way[0]])
        idxs[1] = np.argmax((self.v_list - self.v_list[0]).dot(way))
        idxs[2] = np.argmax(np.abs((self.v_list - self.v_list[0]).dot(norm_way)))
        idxs[3] = np.argmax(-(self.v_list - self.v_list[0]).dot(way))

        return idxs

    def _next_idxs(self, cur_idxs: npt.NDArray):

        v_len = len(self.v_list)
        
        def _next_one(index: int, call_back: tp.Callable):
            cur_val = call_back(self.v_list[index])
            for next_index, v_index in enumerate(range(index+1-v_len,index+1), index):
                val = call_back(self.v_list[v_index])
                if val < cur_val:
                    break
                cur_val = val

            return (next_index) % v_len


        next_idxs = np.zeros_like(cur_idxs)
        way = self._get_support_side(cur_idxs[0]+1)
        norm_way = np.array([way[1], -way[0]])

        next_idxs[0] = (cur_idxs[0] + 1) % v_len
        next_idxs[1] = _next_one(cur_idxs[1], lambda v: (v - self.v_list[cur_idxs[0]]).dot(way))
        next_idxs[2] = _next_one(cur_idxs[2], lambda v: np.abs((v - self.v_list[cur_idxs[0]]).dot(norm_way)))
        next_idxs[3] = _next_one(cur_idxs[3], lambda v: -(v - self.v_list[cur_idxs[0]]).dot(way))

        return next_idxs

    def _get_criterion(self, idxs: npt.NDArray):
        
        support_side = self._get_support_side(idxs[0])
        ss_norm = support_side.dot(support_side)

        if ss_norm == 0:
            raise ValueError("There are identical vertices in the input array")

        norm_support_side = np.array([support_side[1], -support_side[0]])
        
        idxs_vertes = self.v_list[idxs]
        obb_vertes = Poligon._get_obb_vertes(support_side, norm_support_side, idxs_vertes)

        return self.criterion(obb_vertes)

    def _get_support_side(self, index):
        
        v_len = len(self.v_list)

        return self.v_list[index % v_len] - self.v_list[(index-1)%v_len] 

    def _get_obb_vertes(support_side: npt.NDArray, norm_support_side: npt.NDArray, s_vertes: npt.NDArray):
        
        obb_vertes = []
        ss, nss = np.copy(support_side), np.copy(norm_support_side)
        
        for v1, v2 in zip(s_vertes, s_vertes[range(-1, 3)]):
            mA = np.array([nss, ss])
            vb = np.array([nss.dot(v1), ss.dot(v2)])  
            vx = Poligon._solve_Ax_b_2x2(mA, vb)
            obb_vertes.append(vx)
            ss, nss = nss, ss

        return obb_vertes     

    def _det2x2(m: npt.NDArray):
        
        return m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]

    def _solve_Ax_b_2x2(mA: npt.NDArray, vb: npt.NDArray):
        
        mA = mA + fr.Fraction(0)
        vb = vb + fr.Fraction(0)

        mA_det = Poligon._det2x2(mA)
        x = np.zeros_like(vb)
        x[0] = Poligon._det2x2(np.array([vb, mA[:, 1]])) / mA_det
        x[1] = Poligon._det2x2(np.array([mA[:, 0], vb])) / mA_det

        return x



    