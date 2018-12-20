# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:41:55 2017
https://zhuanlan.zhihu.com/p/28690706
@author: Quantum Liu
"""
'''
Example:
gm=GPUManager()
with torch.cuda.device(gm.auto_choice()):
    blabla

Or:
gm=GPUManager()
torch.cuda.set_device(gm.auto_choice())
'''

import os
import torch
import threading
from multiprocessing import Lock
from nise_lib.nise_functions import debug_print


def parse(line, qargs):
    '''
    line:
        a line of text
    qargs:
        query arguments
    return:
        a dict of gpu infos
    Pasing a line of csv format text returned by nvidia-smi
    解析一行nvidia-smi返回的csv格式文本
    '''
    numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']  # 可计数的参数
    power_manage_enable = lambda v: (not 'Not Support' in v)  # lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
    to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').replace('W', ''))  # 带单位字符串去掉单位
    process = lambda k, v: (
        (int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
    return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}


def query_gpu(qargs = (), available_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', default = '')):
    '''
    qargs:
        query arguments
    return:
        a list of dict
    Querying GPUs infos
    查询GPU信息
    '''
    qargs = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']
    qargs.extend(qargs)
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    all_gpus = [parse(line, qargs) for line in results]
    if available_gpu:
        _gpus = []
        available_gpu = available_gpu.split(',')
        for a in available_gpu:
            _gpus.append(all_gpus[int(a)])
        all_gpus = _gpus
    return all_gpus


def by_power(d):
    '''
    helper function fo sorting gpus by power
    '''
    power_infos = (d['power.draw'], d['power.limit'])
    if any(v == 1 for v in power_infos):
        debug_print('Power management unable for GPU {}'.format(d['index']))
        return 1
    return float(d['power.draw']) / d['power.limit']


class GPUManager():
    '''
    qargs:
        query arguments
    A manager which can list all available GPU devices
    and sort them and choice the most free one.Unspecified
    ones pref.
    GPU设备管理器，考虑列举出所有可用GPU设备，并加以排序，自动选出
    最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定，
    优先选择未指定的GPU。
    '''
    
    def __init__(self, qargs = ()):
        '''
        '''
        self.qargs = qargs
        self.gpus = query_gpu(qargs)
        for gpu in self.gpus:
            gpu['specified'] = False
        self.gpu_num = len(self.gpus)
        self.gpu_locks = [Lock() for _ in range(self.gpu_num)]
    
    def _sort_by_memory(self, gpus, by_size = False):
        debug_print('Free memory:', '; '.join(['GPU %s: %s MB' % (g['index'], g['memory.free'])
                                               for g in gpus]))
        if by_size:
            debug_print('Sorted by free memory size')
            return sorted(gpus, key = lambda d: d['memory.free'], reverse = True)
        else:
            debug_print('Sorted by free memory rate')
            return sorted(gpus, key = lambda d: float(d['memory.free']) / d['memory.total'], reverse = True)
    
    def _sort_by_power(self, gpus):
        return sorted(gpus, key = by_power)
    
    def _sort_by_custom(self, gpus, key, reverse = False, qargs = ()):
        if isinstance(key, str) and (key in qargs):
            return sorted(gpus, key = lambda d: d[key], reverse = reverse)
        if isinstance(key, type(lambda a: a)):
            return sorted(gpus, key = key, reverse = reverse)
        raise ValueError(
            "The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")
    
    def get_i_by_index(self, index):
        
        for i, gpu in enumerate(self.gpus):
            if index == gpu['index']:
                return i
    
    def lock_acquire_by_index(self, index):
        i = self.get_i_by_index(index)
        debug_print('GM is', id(self), 'thread', threading.get_ident(), 'acquiring the', i, 'th lock, the index is',
                    index,
                    id(self.gpu_locks[self.get_i_by_index(index)]), )
        self.gpu_locks[i].acquire()
        debug_print('GPU %s\'s lock ACQuired.' % index, 'thread', threading.get_ident(), 'gets lock',
                    id(self.gpu_locks[self.get_i_by_index(index)]))
    
    def lock_release_by_index(self, index):
        i = self.get_i_by_index(index)
        debug_print('GM is', id(self), 'thread', threading.get_ident(), 'releasing the', i, 'th lock, the index is',
                    index,
                    id(self.gpu_locks[self.get_i_by_index(index)]), )
        self.gpu_locks[i].release()
        debug_print('GPU %s\'s lock RELeased.' % index, 'thread', threading.get_ident(), 'releases lock',
                    id(self.gpu_locks[self.get_i_by_index(index)]))
    
    def lock_acquire_by_index_ex(self, gpu_locks, index):
        i = self.get_i_by_index(index)
        debug_print('GM is', id(self), 'thread', threading.get_ident(), 'acquiring the', i, 'th lock, the index is',
                    index,
                    id(self.gpu_locks[self.get_i_by_index(index)]), )
        gpu_locks[i].acquire()
        debug_print('GPU %s\'s lock ACQuired.' % index, 'thread', threading.get_ident(), 'gets lock',
                    id(self.gpu_locks[self.get_i_by_index(index)]))
    
    def lock_release_by_index_ex(self, gpu_locks, index):
        i = self.get_i_by_index(index)
        debug_print('GM is', id(self), 'thread', threading.get_ident(), 'releasing the', i, 'th lock, the index is',
                    index,
                    id(self.gpu_locks[self.get_i_by_index(index)]), )
        gpu_locks[i].release()
        debug_print('GPU %s\'s lock RELeased.' % index, 'thread', threading.get_ident(), 'releases lock',
                    id(self.gpu_locks[self.get_i_by_index(index)]))
    
    def auto_choice(self, mode = 0):
        '''
        mode:
            0:(default)sorted by free memory size
        return:
            a TF device object
        Auto choice the freest GPU device,not specified
        ones
        自动选择最空闲GPU,返回索引
        '''
        debug_print('Choosing GPU ... ')
        for old_infos, new_infos in zip(self.gpus, query_gpu(self.qargs)):
            old_infos.update(new_infos)
        # unspecified_gpus = [gpu for gpu in self.gpus if not gpu['specified']] or self.gpus
        unspecified_gpus = self.gpus
        
        if mode == 0:
            debug_print('Choosing the GPU device has largest free memory...')
            chosen_gpu = self._sort_by_memory(unspecified_gpus, True)[0]
        elif mode == 1:
            debug_print('Choosing the GPU device has highest free memory rate...')
            chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
        elif mode == 2:
            debug_print('Choosing the GPU device by power...')
            chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
        else:
            debug_print('Given an unaviliable mode, will be chosen by memory')
            chosen_gpu = self._sort_by_memory(unspecified_gpus)[0]
        chosen_gpu['specified'] = True
        chosen_index = chosen_gpu['index']
        msg = 'GPU Chosen {i}:\n{info}'.format(i = chosen_index, info = '\n'.join(
            [str(k) + ':' + str(v) for k, v in chosen_gpu.items()]))
        for m in msg.split('\n'): debug_print(m)
        locks_reverse = list(self.gpus)
        locks_reverse.reverse()
        
        return int(chosen_index)


global gm
gm = GPUManager()
