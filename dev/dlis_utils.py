import pandas as pd
from dlisio import dlis

def get_fmi_segment(fname, top, base, channel='FMI_DYN'):
    """
    Returns the depths (1-D array) and FMI image log (2-D array)
    corresponding to the filename (fname)
    """
    physical_file = dlis.load('../data/FMI_Run3_processed.dlis')
    f, *f_tail  = physical_file
    frame = f.frames[0]
    curves = frame.curves()
    fmi = curves[channel]
    depth = curves['TDEP']
    depth_slice = (depth >= top) & (depth <= base)
    depth_seg = depth[depth_slice]
    fmi_seg = fmi[depth_slice, :]
    return depth_seg, fmi_seg



def print_frame_summary(f):
    for i, frame in enumerate(f.frames):
        print('frame ', i)
        index_channel = next(ch for ch in frame.channels if ch.name == frame.index)
        print(f'name                  : {frame.name}')
        print(f'description           : {frame.description}')
        print(f'index_type            : {frame.index_type}')
        print(f'index_min/max         : {frame.index_min}, {frame.index_max}, {index_channel.units}')
        print(f'index_channel.units   : {index_channel.units}')
        print(f'direction             : {frame.direction}')
        print(f'spacing               : {frame.spacing} {index_channel.units}')
        print(f'index_channel         : {index_channel}')
        print(f'No. of channels       : {len(frame.channels)}')
        print()
    return


def summarize(objs, name='Name', long_name='Long name', units='Units',
              dimension='Dimension', **kwargs):
    """Create a pd.DataFrame that summarize the content of 'objs', One 
    object pr. row
    
    Parameters
    ----------
    
    objs : list()
        list of metadata objects
        
    **kwargs
        Keyword arguments 
        Use kwargs to tell summarize() which fields (attributes) of the 
        objects you want to include in the DataFrame. The parameter name 
        must match an attribute on the object in 'objs', while the value 
        of the parameters is used as a column name. Any kwargs are excepted, 
        but if the object does not have the requested attribute, 'KeyError' 
        is used as the value.
        
    Returns
    -------
    
    summary : pd.DataFrame
    """
    default_kwargs = {'name': name,
                     'long_name': long_name,
                     'units': units,
                     'dimension': dimension,
                     }

    kwargs.update(default_kwargs)

    summary = []
    for attr, label in kwargs.items():
        column = []
        for obj in objs:
            try:
                value = getattr(obj, attr)
            except AttributeError:
                value = 'KeyError'
    
            column.append(value)
        summary.append(column)

    summary = pd.DataFrame(summary).T
    summary.columns = kwargs.values()
    return summary
