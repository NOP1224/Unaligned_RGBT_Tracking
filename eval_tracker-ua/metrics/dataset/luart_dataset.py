
from .basedataset import BaseRGBTDataet,_basepath
from metrics.utils import *
import os
from metrics.metrics.metrics_luart import PR_LUART, SR_LUART, NPR_LUART

class LUART(BaseRGBTDataet):
    """
    Publication: `LUART: A Large-scale High-diversity Benchmark for RGBT Tracking`
    """
    def __init__(self, gt_path=f'/data1/Code/jinjiandong/unalignedtracking-baseline/eval_tracker/metrics/gt_file/LUART/luart_gt/rgb',
                 seq_name_path=f"/data1/Code/jinjiandong/unalignedtracking-baseline/eval_tracker/metrics/gt_file/LUART/testing_list.txt") -> None:
        seqs = load_text(seq_name_path, dtype=str)
        super().__init__(gt_path=gt_path, seqs=seqs, bbox_type='ltwh')

        self.name = 'LUART_test'
        self.PR_fun = PR_LUART()
        self.SR_fun = SR_LUART()
        self.NPR_fun = NPR_LUART()

        # Challenge attributes
        self._attr_list = ('PO', 'TO','LI', 'HI', 'IV','DEF', 
                            'BC','MB','FM','SV','ARC','SA',
                            'TC','FL','SO','TOV','HM','VM',
                            'FR','PAC','FG','CZ')
        
        self.PO = self.choose_serial_by_att('PO')
        self.TO = self.choose_serial_by_att('TO')
        self.LI = self.choose_serial_by_att('LI')
        self.HI = self.choose_serial_by_att('HI')
        self.IV = self.choose_serial_by_att('IV')
        self.DEF = self.choose_serial_by_att('DEF')


        self.TC = self.choose_serial_by_att('TC')
        self.SO = self.choose_serial_by_att('SO')
        self.FM = self.choose_serial_by_att('FM')
        self.BC = self.choose_serial_by_att('BC')
        
        self.ARC = self.choose_serial_by_att('ARC')        
        self.MB = self.choose_serial_by_att('MB')
        self.SV = self.choose_serial_by_att('SV')
        self.SA = self.choose_serial_by_att('SA')
        self.FL = self.choose_serial_by_att('FL')
         
        self.FG = self.choose_serial_by_att('FG')
        self.FR = self.choose_serial_by_att('FR')
        self.PAC = self.choose_serial_by_att('PAC')
        self.TOV = self.choose_serial_by_att('TOV')
        self.VM = self.choose_serial_by_att('VM')
        self.HM = self.choose_serial_by_att('HM')
        

    def get_attr_list(self):
        return self._attr_list

    def choose_serial_by_att(self, attr):
        if attr==self.ALL:
            return self.seqs_name
        else:
            seqs = []
            for seq in self.seqs_name:
                i = self.get_attr_list().index(attr)
                path = os.path.join(self.gt_path, '..', '..', 'attribute_annotation', seq+'.txt')
                p = load_text(path)[i]
                if p==1.:
                    seqs.append(seq)
            return seqs

    def PR(self, tracker_name=None, seqs=None):
        """
        Parameters
        ----------
        [in] tracker_name - str
            Default is None, evaluate all registered trackers.
        [in] seqs - list
            Sequence to be evaluated, default is all.
        
        Returns
        -------
        [out0] When evaluating a single tracker, return MPR and the precision Rate at different thresholds.
        [out1] Other cases return a dictionary with all tracker results.
        """
        if seqs==None:
            seqs = self.seqs_name

        if tracker_name!=None:
            return self.PR_fun(self, self.trackers[tracker_name], seqs)
        else:
            res = {}
            for k,v in self.trackers.items():
                res[k] = self.PR_fun(self, v, seqs)
            return res

    def NPR(self, tracker_name=None, seqs=None):
        """
        """
        if seqs==None:
            seqs = self.seqs_name

        if tracker_name!=None:
            return self.NPR_fun(self, self.trackers[tracker_name], seqs)
        else:
            res = {}
            for k,v in self.trackers.items():
                res[k] = self.NPR_fun(self, v, seqs)
            return res


    def SR(self, tracker_name=None, seqs=None):
        """
        Parameters
        ----------
        [in] tracker_name - str
            Default is None, evaluate all registered trackers.
        [in] seqs - list
            Sequence to be evaluated, default is all.
        """
        if seqs==None:
            seqs = self.seqs_name

        if tracker_name!=None:
            return self.SR_fun(self, self.trackers[tracker_name], seqs)
        else:
            res = {}
            for k,v in self.trackers.items():
                res[k] = self.SR_fun(self, v, seqs)
            return res


    def draw_attributeRadar(self, metric_fun, filename=None):
        if filename==None:
            filename = self.name
            if metric_fun==self.PR:
                filename+="_PR"
            elif metric_fun==self.SR:
                filename+="_SR"
            filename+="_radar.png"
        return super().draw_attributeRadar(metric_fun, filename)
    

    def draw_plot(self, metric_fun, filename=None, title=None, seqs=None):
        assert metric_fun in [self.NPR, self.PR, self.SR]
        if filename==None:
            filename = self.name
            if metric_fun==self.PR:
                filename+="_PR"
                axis = self.PR_fun.thr
                loc = "lower right"
                x_label = "Location error threshold"
                y_label = "Precision"
            elif metric_fun==self.NPR:
                filename+="_NPR"
                axis = self.NPR_fun.thr
                loc = "lower right"
                x_label = "Normalized Location error threshold"
                y_label = "Normalized Precision"
            elif metric_fun==self.SR:
                filename+="_SR"
                axis = self.SR_fun.thr
                loc = "lower left"
                x_label = "Overlap threshold"
                y_label = "Success Rate"
            filename+="_plot.png"

        if title==None:
            if metric_fun==self.PR:
                title="Precision plots of OPE on LUART"
            elif metric_fun==self.NPR:
                title="Normalized Precision plots of OPE on LUART"
            elif metric_fun==self.SR:
                title="Success plots of OPE on LUART"

        return super().draw_plot(axis=axis, 
                                 metric_fun=metric_fun, 
                                 filename=filename, 
                                 title=title, 
                                 seqs=seqs, y_max=1.0, y_min=0.0, loc=loc,
                                 x_label=x_label, y_label=y_label)