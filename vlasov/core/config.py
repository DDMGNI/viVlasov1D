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
        self.valid     = self.validate(self.validator, copy=True)
        
    
    def write_default_config(self):
        '''
        Reads default values from runspec file and creates a default
        configuration file in run.cfg.default.
        '''
        
        self.write()
        
    
    def write_current_config(self, filename):
        tmp = self.filename
        self.filename = filename
        self.write()
        self.filename = tmp
    
    
    def is_timestepping_mp(self):
        return self['solver']['timestepping'].lower() == 'mp'
    
    def is_timestepping_rk2(self):
        return self['solver']['timestepping'].lower() == 'rk2'
    
    def is_timestepping_rk4(self):
        return self['solver']['timestepping'].lower() == 'rk4'
    
    
    def is_poisson_bracket_arakawa_J1(self):
        return self['solver']['poisson_bracket'].lower() == 'arakawaj1'
    
    def is_poisson_bracket_arakawa_J2(self):
        return self['solver']['poisson_bracket'].lower() == 'arakawaj2'
    
    def is_poisson_bracket_arakawa_J4(self):
        return self['solver']['poisson_bracket'].lower() == 'arakawaj4'
    
    def is_poisson_bracket_simpson(self):
        return self['solver']['poisson_bracket'].lower() == 'simpson'
    
    
    def is_laplace_operator_CFD2(self):
        return self['solver']['laplace_operator'].lower() == 'cfd2'
    
    def is_laplace_operator_CFD4(self):
        return self['solver']['laplace_operator'].lower() == 'cfd4'
    
    def is_laplace_operator_simpson(self):
        return self['solver']['laplace_operator'].lower() == 'simpson'
    

    def is_averaging_operator_none(self):
        return self['solver']['averaging_operator'] == None or self['solver']['averaging_operator'].lower() == 'none'
    
    def is_averaging_operator_midpoint(self):
        return self['solver']['averaging_operator'].lower() == 'midpoint'
    
    def is_averaging_operator_simpson(self):
        return self['solver']['averaging_operator'].lower() == 'simpson'
    
    def is_averaging_operator_arakawa_J1(self):
        return self['solver']['averaging_operator'].lower() == 'arakawaj1'
    
    def is_averaging_operator_arakawa_J2(self):
        return self['solver']['averaging_operator'].lower() == 'arakawaj2'
    
    def is_averaging_operator_arakawa_J4(self):
        return self['solver']['averaging_operator'].lower() == 'arakawaj4'
    
    
    def is_dissipation_none(self):
        if self['solver']['dissipation'] == None or self['solver']['coll_freq'] == 0.:
            return True
        else:
            return self['solver']['dissipation'].lower() == 'none'
    
    def is_dissipation_collisions(self):
        if self['solver']['dissipation'] == None or self['solver']['coll_freq'] == 0.:
            return False
        else:
            return self['solver']['dissipation'].lower() == 'collisions'
    
    def is_dissipation_double_bracket(self):
        if self['solver']['dissipation'] == None or self['solver']['coll_freq'] == 0.:
            return False
        else:
            return self['solver']['dissipation'].lower() == 'double_bracket'
    


if __name__ == '__main__':
    '''
    Instantiates a Config object and creates a default configuration file.
    '''
    
    filename = 'run.cfg.default'
    
    if os.path.exists(filename):
        os.remove(filename)
    
    config = Config(filename, file_error=False)
    config.write_default_config()

