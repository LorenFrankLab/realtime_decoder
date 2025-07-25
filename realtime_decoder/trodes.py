import numpy as np


def _extract_repeats(dtype_info):
    
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

def _get_dtype_info(header, tag):

    name, dtype_info = tuple(tag.split(' '))
    if '*' in tag:
        repeats, dtype = _extract_repeats(dtype_info)
    else:
        repeats = 0
        dtype = dtype_info

    if not getattr(np, dtype):
        raise ValueError(f"Unrecognized datatype {dtype}")

    try:
        endian = header['Byte_order']
    except KeyError:
        endian = 'little endian'

    if endian == 'little endian':
        endian_str = '<'
    else:
        # Currently not aware of any Trodes file that is not little endian
        raise ValueError(
            f"Endianess is {endian}, are you sure this is a file Trodes file?")

    if dtype == 'double':
        symbol = 'f'
        num_bytes = 8
    elif dtype == 'float':
        symbol = 'f'
        num_bytes = 4
    else:
        if not ('i' in dtype or 'u' in dtype):
            raise ValueError(f"Unrecognized dtype {dtype}")
        symbol = dtype[0]
        num_bytes = np.dtype(dtype).itemsize

    dt = endian_str + f'{symbol}{num_bytes}'
    
    return name, dt, repeats

def _get_tag_locs(fields_info):
    
    starts = []
    stops = []
    for ii, char in enumerate(fields_info):
        if char == '<':
            starts.append(ii+1)
        elif char == '>':
            stops.append(ii)
            
    if len(starts) != len(stops):
        raise ValueError(
            f"Could not extract datatype info from line {fields_info}")
        
    return starts, stops


def _get_numpy_dtype(header):
    
    fields_info = header['Fields']
    
    starts, stops = _get_tag_locs(fields_info)

    dtype_list = []
    for start, stop in zip(starts, stops):
        tag = fields_info[start:stop]
        name, dt, repeats = _get_dtype_info(header, tag)
        
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

def _extract_header(l):
    
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


def _verify_header(header, keys):
    
    for key in keys:
        if not key in header:
            raise ValueError(
                f"Could not find '{key}' in header!"
            )
            
def _verify_dtype(numpy_dt, keys):

    fields = list(numpy_dt.fields)
    for key in keys:
        matches = [field for field in fields if field.find(key) >= 0]
        if matches == []:
            raise ValueError(
                f"Could not find field '{key}' in "
                f"the datatype {fields}"
            )

def load_data_file(file, *, header_keys=[], dtype_fields=[]):

    # header keys: keys in the header section that should be checked
    # for their existence
    # dtype fields: names that are expected to be in the 'Fields'
    # section in the header section

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
            
        dtype = _get_numpy_dtype(header)
        if dtype_fields != []:
            _verify_dtype(dtype, dtype_fields)

        data = np.fromfile(f, dtype=dtype)

    return data, header