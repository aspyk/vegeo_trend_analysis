import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri as rpyn
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np


class Rwrapper():

    def __init__(self):
        pass

    def install_package(self, pname):
        print('Check package {}...'.format(pname))
        # import R's utility package
        utils = rpackages.importr('utils')

        # select a mirror for R packages
        utils.chooseCRANmirror(ind=1) # select the first mirror in the list

        # Selectively install what needs to be install.
        # We are fancy, just because we can.
        if not rpackages.isinstalled(pname):
            print('Package {} not installed, install it...'.format(pname))
            utils.install_packages(pname)
            print('Package {} installed.'.format(pname))
        else:
            print('Package {} already installed.'.format(pname))

    def test1(self):

        rkt = rpackages.importr('rkt')

        nyear = 4
        nseas = 5
        year = np.repeat(np.arange(2000,2000+nyear), nseas)
        dekad = np.tile(1+np.arange(nseas), nyear)
        data = np.random.rand(nseas*nyear) + np.arange(nseas*nyear)*0.1

        if 1:
            year = robjects.IntVector(year)
            dekad = robjects.IntVector(dekad)
            data = robjects.FloatVector(data)
        else:
            year = rpyn.numpy2ri(year)
            dekad = rpyn.numpy2ri(dekad)
            data = rpyn.numpy2ri(data)

        print(year)
        print(dekad)
        print(data)
       
        self.res = rkt.rkt(year, data, dekad)
        print(self.res)

        df = pandas2ri.ri2py_dataframe(rw.res).transpose()
        df.columns = self.res.names
        df = df[['sl', 'S', 'B', 'varS', 'tau']]

        print(pd.concat([df,df,df]))
        self.df =df

    def test2(self):

        rkt = rpackages.importr('rkt')

        nyear = 4
        nseas = 5
        year = np.repeat(np.arange(2000,2000+nyear), nseas)
        dekad = np.tile(1+np.arange(nseas), nyear)
        data = np.random.rand(nseas*nyear) + np.arange(nseas*nyear)*0.1

        res = self.smk(year, data, dekad)
        print(res)
 
    def smk(self, year, data, block=None):

        rkt = rpackages.importr('rkt')

        year = robjects.IntVector(year)
        data = robjects.FloatVector(data)
        if block is not None:
            block = robjects.IntVector(block)
            self.res = rkt.rkt(year, data, block)
        else:
            self.res = rkt.rkt(year, data)

        print(self.res)

        df = pandas2ri.ri2py_dataframe(self.res).transpose()
        df.columns = self.res.names
        df = df[['sl', 'S', 'B', 'varS', 'tau']]

        return df

    def MannKendall(self, data):

        rkt = rpackages.importr('Kendall')

        data = robjects.FloatVector(data)

        self.res = rkt.MannKendall(data)
        print(self.res)

        df = pandas2ri.ri2py_dataframe(self.res).transpose()
        df.columns = self.res.names
        df = df[['sl', 'S', 'B', 'varS', 'tau']]

        return df


if __name__=='__main__':
    
    rw = Rwrapper()
    
    rw.install_package('rkt')

    rw.test2()



    
