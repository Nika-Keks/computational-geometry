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

from numpy import typing as npt


__all__ = ["Poligon"]

class Poligon:

    def __init__(self, vertex_list: np.ndarray, criterion = lambda a, b: a + b) -> None:
        
        self.v_list = vertex_list.astype(np.int64)
        self.criterion = criterion


    def get_obb(self):
        
        start_idxs = self._get_start_idxs()
        min_idxs = np.copy(start_idxs)
        min_cval = self._get_criterion(min_idxs)
        cur_idxs = self._next_idxs(start_idxs)

        while cur_idxs[0] != start_idxs[0]:
            cur_cval = self._get_criterion(cur_idxs)

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
            for next_index, val in enumerate(map(call_back, self.v_list[range(index+1-v_len,index+1)]), index):
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

        support_side = support_side.astype(np.float) / np.sqrt(float(ss_norm))
        norm_support_side = np.array([support_side[1], -support_side[0]])

        width_d = self.v_list[idxs[1]] - self.v_list[idxs[3]]
        hight_d = self.v_list[idxs[2]] - self.v_list[idxs[0] - 1]

        a = np.abs(hight_d.dot(norm_support_side))
        b = np.abs(width_d.dot(support_side))
        
        return self.criterion(a, b)

    def _get_support_side(self, index):
        v_len = len(self.v_list)

        return self.v_list[index % v_len] - self.v_list[(index-1)%v_len]
        