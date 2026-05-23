
from .basedataset import BaseRGBTDataet,_basepath
from rgbt.utils import *
import os
from rgbt.metrics import PR_LasHeR,SR_LasHeR,NPR

class MUART244(BaseRGBTDataet):
    def __init__(self, gt_path=f'/data1/Code/jinjiandong/Neuro_Unaligned-0925/eval_tracker/metrics/gt_file/MUART244/annos/visible',
                 seq_name_path=f"/data1/Code/jinjiandong/Neuro_Unaligned-0925/eval_tracker/metrics/gt_file/MUART244/testinglist.txt") -> None:
        seqs = load_text(seq_name_path, dtype=str)
        super().__init__(gt_path=gt_path, seqs=seqs, bbox_type='ltwh')

        self.name = 'MUART244'
        self.PR_fun = PR_LasHeR()
        self.SR_fun = SR_LasHeR()
        self.NPR_fun = NPR()

        # Challenge attributes
        self._attr_list = (
            'NO', 'PO', 'TO', 'OV', 'TOV',
            'LI', 'HI', 'AIV', 'LR',
            'BC', 'SA', 'TC', 'CM',
            'FM', 'SV', 'ARC',
            'HM', 'VM', 'RM', 'PAC', 'FG', 'CZ',
            'HD', 'UAV',

            # ==== 新增 OFF1 ~ OFF11 ====
            'OFF1','OFF2','OFF3','OFF4','OFF5',
            'OFF6','OFF7','OFF8','OFF9','OFF10','OFF11',

            # ==== 新增 AREA1 ~ AREA8 ====
            'AREA1','AREA2','AREA3','AREA4',
            'AREA5','AREA6','AREA7','AREA8'
        )

        # ========================
        # 原有挑战类型
        # ========================
        self.NO  = self.choose_serial_by_att('NO')
        self.PO  = self.choose_serial_by_att('PO')
        self.TO  = self.choose_serial_by_att('TO')
        self.OV  = self.choose_serial_by_att('OV')
        self.TOV = self.choose_serial_by_att('TOV')

        self.LI  = self.choose_serial_by_att('LI')
        self.HI  = self.choose_serial_by_att('HI')
        self.AIV = self.choose_serial_by_att('AIV')
        self.LR  = self.choose_serial_by_att('LR')

        self.BC  = self.choose_serial_by_att('BC')
        self.SA  = self.choose_serial_by_att('SA')
        self.TC  = self.choose_serial_by_att('TC')
        self.CM  = self.choose_serial_by_att('CM')

        self.FM  = self.choose_serial_by_att('FM')
        self.SV  = self.choose_serial_by_att('SV')
        self.ARC = self.choose_serial_by_att('ARC')

        self.HM  = self.choose_serial_by_att('HM')
        self.VM  = self.choose_serial_by_att('VM')
        self.RM  = self.choose_serial_by_att('RM')
        self.PAC = self.choose_serial_by_att('PAC')
        self.FG  = self.choose_serial_by_att('FG')
        self.CZ  = self.choose_serial_by_att('CZ')
        self.HD  = self.choose_serial_by_att('HD')
        self.UAV = self.choose_serial_by_att('UAV')

        # ========================
        # 新增偏移 OFF1~OFF11
        # ========================
        self.OFF1  = self.choose_serial_by_att('OFF1')
        self.OFF2  = self.choose_serial_by_att('OFF2')
        self.OFF3  = self.choose_serial_by_att('OFF3')
        self.OFF4  = self.choose_serial_by_att('OFF4')
        self.OFF5  = self.choose_serial_by_att('OFF5')
        self.OFF6  = self.choose_serial_by_att('OFF6')
        self.OFF7  = self.choose_serial_by_att('OFF7')
        self.OFF8  = self.choose_serial_by_att('OFF8')
        self.OFF9  = self.choose_serial_by_att('OFF9')
        self.OFF10 = self.choose_serial_by_att('OFF10')
        self.OFF11 = self.choose_serial_by_att('OFF11')

        # ========================
        # 新增尺度 AREA1~AREA8
        # ========================
        self.AREA1 = self.choose_serial_by_att('AREA1')
        self.AREA2 = self.choose_serial_by_att('AREA2')
        self.AREA3 = self.choose_serial_by_att('AREA3')
        self.AREA4 = self.choose_serial_by_att('AREA4')
        self.AREA5 = self.choose_serial_by_att('AREA5')
        self.AREA6 = self.choose_serial_by_att('AREA6')
        self.AREA7 = self.choose_serial_by_att('AREA7')
        self.AREA8 = self.choose_serial_by_att('AREA8')

    def get_attr_list(self):
        return self._attr_list

    def choose_serial_by_att(self, attr):
        if attr==self.ALL:
            return self.seqs_name
        else:
            seqs = []
            for seq in self.seqs_name:
                i = self.get_attr_list().index(attr)
                path = os.path.join(self.gt_path, '..', '..', 'AttriSeqsTxt', seq+'.txt')
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
                title="Precision plots of OPE on MUART244"
            elif metric_fun==self.NPR:
                title="Normalized Precision plots of OPE on MUART244"
            elif metric_fun==self.SR:
                title="Success plots of OPE on MUART244"

        return super().draw_plot(axis=axis, 
                                 metric_fun=metric_fun, 
                                 filename=filename, 
                                 title=title, 
                                 seqs=seqs, y_max=1.0, y_min=0.0, loc=loc,
                                 x_label=x_label, y_label=y_label)