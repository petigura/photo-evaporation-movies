import os 
import pandas as pd
import glob
import numpy as np

names = 'time radius_core radius_ratio per'.split()
DATADIR = os.path.join(os.path.dirname(__file__),'../data/')
CACHEFN = os.path.join(DATADIR,'load_table_cache.hdf')

def load_table(table,cache=0, cachefn=CACHEFN):
    """
    First time through, run with cache=2
    """
    if cache==1:
        try:
            df = pd.read_hdf(cachefn,table)
            print "read table {} from {}".format(table,cachefn)
            df['radius_total'] = df.radius_core * (1 + df.radius_ratio) # computed on the fly
            return df
        except IOError:
            print "Could not find cache file: %s" % cachefn
            print "Building cache..."
            cache=2
        except KeyError:
            print "Cache not built for table: %s" % table
            print "Building cache..."
            cache=2

    if cache==2:
        cmd = 'tar -xzf {}/data.tar.gz'.format(DATADIR)
        os.system(cmd)
        df = load_table(table, cache=False)
        print "writing table {} to cache".format(table)
        df.to_hdf(cachefn,table,complevel=1,complib='blosc')
        cmd = 'rm -R {}/vid_data'.format(DATADIR)
        os.system(cmd)
        return df

    if table=='energy-limited':
        fL = glob.glob('vid_data/out_for_vid_0*.dat')
    elif table=='variable-efficiency':
        fL = glob.glob('vid_data/out_for_vid_2*.dat')
    else:
        assert False, "mode is wrong"

    names = 'time radius_core radius_ratio per'.split()
    df = []
    for i,f in enumerate(fL):
        if i % 1000 == 0:
            print i
        _df = pd.read_table(f,names=names,sep='\s+')
        _df['iplanet'] = i
        _df['itime'] = np.arange(len(_df))
        df.append(_df)

    df = pd.concat(df)
    return df
  
