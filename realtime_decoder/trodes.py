import numpy as np

# import os
# import glob
# import time

from typing import List, Dict, Tuple
# import logging
# import logging.config

# logger = logging.getLogger("trodes")

# from collections import OrderedDict

# from realtime_decoder import base, messages, datatypes, utils
# from realtime_decoder.datatypes import Datatypes

def _find_tag_bounds(s:str) -> Tuple[List[int], List[int]]:
    
    starts = []
    stops = []
    for ii, char in enumerate(s):
        if char == '<':
            starts.append(ii)
        elif char == '>':
            stops.append(ii)

    if len(starts) != len(stops):
        raise ValueError(
            f"Error finding tag bounds from line {s}")
        
    return starts, stops

def _extract_tag_content(s:str) -> List[str]:

    # input data is string of form <*><*>...
    # where * can be any string and there can be
    # arbitrary number of tags
    
    tag_contents = []
    starts, stops = _find_tag_bounds(s)
    
    for (start, stop) in zip(starts, stops):
        # start and stop are indices of '<' and '>' characters.
        # we need to index as below to capture the content
        # in between the brace characters
        tag_contents.append(s[start+1:stop])

    return tag_contents

def _extract_repeats(dtype_info:str) -> Tuple[int, str]:
    
    # can be num*dtype or dtype*num
    a, b = tuple(dtype_info.split("*"))
    if a.isdigit():
        num_repeats = int(a)
        dtype = b
    elif b.isdigit():
        num_repeats = int(b)
        dtype = a
    else:
        raise ValueError(
            f"Could not parse datatype info {dtype_info} correctly"
        )
        
    return num_repeats, dtype

def _extract_dtype_info(
    header:Dict[str, str], tag_content:str
) -> Tuple[str, np.dtype, int]:

    name, dtype_info = tuple(tag_content.split(' '))
    if '*' in tag_content:
        repeats, dtype = _extract_repeats(dtype_info)
    else:
        repeats = 0
        dtype = dtype_info

    if not getattr(np, dtype):
        raise ValueError(f"Unrecognized datatype {dtype}")

    try:
        endian = header['Byte_order']
    except KeyError:
        # logger.info("Assuming little endian like most Trodes files are")
        endian = 'little endian'

    if endian == 'little endian':
        endian_str = '<'
    else:
        endian_str = '>'

    if dtype == 'double':
        symbol = 'f'
        num_bytes = 8
    elif dtype == 'float':
        symbol = 'f'
        num_bytes = 4
    else: # must be int or uint
        if not ('i' in dtype or 'u' in dtype):
            raise ValueError(f"Unrecognized dtype {dtype}")
        symbol = dtype[0] # will be either 'i' or 'u'
        num_bytes = np.dtype(dtype).itemsize

    dt = endian_str + f'{symbol}{num_bytes}'
    
    return name, dt, repeats


def _extract_numpy_dtype(header:Dict[str, str]) -> np.dtype:

    dtype_tag_contents = _extract_tag_content(header['Fields'])

    dtype_list = []
    for tag_content in dtype_tag_contents:
        name, dt, repeats = _extract_dtype_info(header, tag_content)
        
        if repeats > 0:
            dtype_list.append((name, dt, (repeats, )))
        else:
            dtype_list.append((name, dt))

    try:
        np_dtype = np.dtype(dtype_list)
    except:
        raise ValueError(
            f"Could not convert {dtype_list} into a numpy datatype "
            "successfully!"
        )
        
    return np_dtype

    # starts, stops = _extract_tag_bounds(fields_info)

    # dtype_list = []
    # for start, stop in zip(starts, stops):
    #     # start and stop are indices of '<' and '>' characters
    #     # we need to index as below to capture the content
    #     # in between the brace characters
    #     tag_content = fields_info[start+1:stop+1]
    #     name, dt, repeats = _extract_dtype_info(header, tag_content)
        
    #     if repeats > 0:
    #         dtype_list.append((name, dt, (repeats, )))
    #     else:
    #         dtype_list.append((name, dt))
        
    # try:
    #     np_dtype = np.dtype(dtype_list)
    # except:
    #     raise ValueError(
    #         f"Could not convert {dtype_list} into a numpy datatype "
    #         "successfully!"
    #     )
        
    # return np_dtype

def _extract_header(l:List[str]) -> Dict[str, str]:
    
    # split a line consisting of the form "key: value" into
    # dictionary
    header_dict = {}
    for item in l:
        loc = item.find(':')
        if loc >= 0:
            key = item[:loc]
            value = item[loc+1:].lstrip(' ')
            header_dict[key] = value
                
    return header_dict


def _verify_header(header:Dict[str, str], keys:List[str]) -> None:
    
    for key in keys:
        if not key in header:
            raise ValueError(
                f"Could not find '{key}' in header!"
            )

def _verify_dtype(numpy_dt:np.dtype, keys:List[str]) -> None:

    fields = list(numpy_dt.fields)
    for key in keys:
        matches = [field for field in fields if field.find(key) >= 0]
        if len(matches) != 1:
            raise ValueError(
                f"Could not find unique field '{key}' in "
                f"the datatype {fields}"
            )

def load_dat_file(file:str, *,
    maxlines:int=500, header_keys:List[str]=[], dtype_fields:List[str]=[]
) -> Tuple[np.array, Dict[str, str]]:

    # header keys: keys in the header section that should be checked
    # for their existence
    # dtype fields: names that are expected to be in the 'Fields'
    # part in the header section

    maxlines = 500
    ct = 0
    header_info = []
    with open(file, 'rb') as f:

        instr = b""
        while instr != b"<End settings>\n":
            ct += 1
            instr = f.readline()
            header_info.append(instr.decode('ascii').strip())

            if ct > maxlines:
                raise ValueError("Could not find header info!")
                
        header = _extract_header(header_info)
        if header_keys != []:
            _verify_header(header, header_keys)
            
        dtype = _extract_numpy_dtype(header)
        if dtype_fields != []:
            _verify_dtype(dtype, dtype_fields)

        data = np.fromfile(f, dtype=dtype)

    return data, header