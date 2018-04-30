from __future__ import print_function

class LogfileParser(object):
    '''A for-me-only class for sorting out the jumbled mess that is output by multiprocessed data with processCcd.py'''
    
    def __init__(self, logfile):
        # all variables must be declared here in order to be class variables
        # default values go here, all can be overridden
        self.__dict__['logfile'] = logfile
        self.__dict__['ignore_lines_starts'] = ['root INFO:',
                                                'CameraMapper INFO:',
                                                'WARNING: You are using',
                                                'specified the number',
                                                'OPENBLAS_NUM_THREADS',
                                                'This may indicate th',
                                                'cause problems. WE H',
                                                'what you are doing a',
                                                'variable LSST_ALLOW_',]
        self.__dict__['ignore_substrings'] = ['Running',
                                              'CameraMapper',]
        self.__dict__['parsed_log'] = {}                                   

    def __setattr__(self, attribute, value):
        if not attribute in self.__dict__:
            print("Cannot set %s" % attribute)
        else:
            self.__dict__[attribute] = value

    def ParseLogfile(self, rewrite_as = ''):
        with open(self.logfile) as f:
            lines = f.readlines()
        
        discarded_lines = []
        for line in lines:
            if any([line.startswith(l) for l in self.ignore_lines_starts]): continue
            if any([line.find(l)!=-1 for l in self.ignore_substrings]): continue

            temp, visit, message, text, pyfile, loglevel, namespace = None, None, None, None, None, None, None
            try:
                temp = line.split('visit\': ')[1]
                visit = temp.split('},')[0]
                message = ' '.join(_ for _ in line.split('- ')[1:])[:-1] #remove newline
                text = line.split('},')[1][1:][13:] # [13:] removes 'tag=set([])))''
                pyfile = text.split('- ')[0][1:-1]
                loglevel = line.split('  ')[0]
                namespace = line.split(' ')[3]
            except:
                discarded_lines.append(line)
            if visit and message and pyfile and loglevel and namespace:
                to_append = '%s %s %s %s'%(loglevel, namespace, pyfile, message)
                if visit in self.parsed_log:
                    self.parsed_log[visit].append(to_append)
                else:
                    self.parsed_log[visit] = []
                    self.parsed_log[visit].append(to_append)
                
        if rewrite_as != '': self._RewriteLogfileInOrder(rewrite_as)
        # return _reParseLogfile(ret), discarded_lines
        return discarded_lines

    def ShowLog(self):
        for key in self.parsed_log.keys():
            for line in self.parsed_log[key]:
                print(key + ': ' + line)
            print()

    def _RewriteLogfileInOrder(self, output_filename):
        print('Rewrote?')
        
        with open(output_filename, mode='w') as f:
            for key in self.parsed_log.keys():
                f.write(key + ': ' + line + '\n')
            f.write('\n')








    def ShowLog(self):
        for key in self.parsed_log.keys():
            for line in self.parsed_log[key]:
                f.write(key + ': ' + line + '\n')
            f.write('\n')

    def _RewriteLogfileInOrder(self, output_filename):
        with open(output_filename, mode='w') as f:
            for key in self.parsed_log.keys():
                for line in self.parsed_log[key]:
                    print(key + ': ' + line)
                print()
        

    # def _RewriteLogfileInOrder(self, output_filename):
    #     for key in self.parsed_log.keys():
    #         for line in self.parsed_log[key]:
    #             f.write(key + ': ' + line + '\n')
    #         f.write('\n')

    # def ShowLog(self):
    #     with open(output_filename, mode='w') as f:
    #         for key in self.parsed_log.keys():
    #             for line in self.parsed_log[key]:
    #                 print key + ': ' + line
    #             print

    def _reParseLogfile(info_dict):
        import re
        id_attribs = ['basename','object','visit','filter','date','expTime']
        ret = {}
        ret['meta'] = {}
        for key in info_dict:
            ret[key] = {}
            for attrib in id_attribs:
                if key.find(attrib)!=-1:
                    ret[key][attrib] = re.sub('\'', '', key.split(attrib)[1][3:].split(',')[0])  
            ret[key]['ALL_OUTPUT'] = info_dict[key]
            
            for linenum, line in enumerate(info_dict[key]):
                if line.find('Magnitude zero point:')!=-1:
                    ret[key]['PHOTOMETRY'] = line.split('Magnitude zero point:')[1].rstrip()
                
                if line.find('Astrometric scatter:')!=-1:
                    ret[key]['ASTROMETRY'] = line.split('Astrometric scatter:')[1].rstrip()  
                
                if line.find('FATAL')!=-1:
                    last_line = info_dict[key][linenum-1].rstrip()
                    fatal_line = info_dict[key][linenum].rstrip()

                    failure_mode = last_line.split(':')[0]
                    if failure_mode not in ret['meta'].keys():
                        ret['meta'][failure_mode] = 1
                    else:
                        ret['meta'][failure_mode] += 1
                    
                    ret[key]['PASSED'] = False
                    ret[key]['OUTCOME'] = 'FAILURE'
                    ret[key]['LINE_ABOVE_FATAL'] = last_line
                    ret[key]['FATAL_LINE'] = fatal_line

                    if last_line.find('processCcd.isr: Performing ISR on sensor')!=-1:
                        ret[key]['FAILURE_MODE'] = 'ISR failure'
                    elif last_line.find('meas.algorithms.psfDeterminer')!=-1 or last_line.find('processCcd.charImage.measurePsf') !=-1:
                        ret[key]['FAILURE_MODE'] = 'PSF failure'
                    elif last_line.find('processCcd.charImage:')!=-1:
                        ret[key]['FAILURE_MODE'] = 'charImage failure'
                    elif last_line.find('detectAndMeasure.measureApCorr:')!=-1:
                        ret[key]['FAILURE_MODE'] = 'measureApCorr failure'                    
                    else:
                        ret[key]['FAILURE_MODE'] = 'other'                    
                    break
                else:
                    ret[key]['PASSED'] = True
                    ret[key]['OUTCOME'] = 'SUCCESS'
        return ret











