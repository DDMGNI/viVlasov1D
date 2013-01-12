'''
Created on Mar 20, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from configobj import ConfigObj
from validate  import Validator

#import numpy as np
#import os
import os.path


class Config(ConfigObj):
    '''
    Run configuration.
    '''


    def __init__(self, infile, file_error=True):
        '''
        Constructor
        '''
        
        self.runspec = 'runspec.cfg'
        
        ConfigObj.__init__(self, infile=infile, configspec=self.runspec, file_error=file_error)
        
        self.validator = Validator()
#        self.valid     = self.validate(self.validator)
        self.valid     = self.validate(self.validator, copy=True)
        
    
    def write_default_config(self):
        '''
        Reads default values from runspec file and creates a default
        configuration file in run.cfg.default.
        '''
        
#        self.validate(self.validator, copy=True)
        self.write()
        
    

if __name__ == '__main__':
    '''
    Instantiates a Config object and creates a default configuration file.
    '''
    
    filename = 'run.cfg.default'
    
    if os.path.exists(filename):
        os.remove(filename)
    
    config = Config(filename, file_error=False)
    config.write_default_config()

