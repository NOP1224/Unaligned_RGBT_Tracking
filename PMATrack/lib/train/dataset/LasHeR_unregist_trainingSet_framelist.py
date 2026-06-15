import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict

from lib.train.data.image_loader import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from lib.train.admin.environment import env_settings
from lib.train.dataset.depth_utils import get_x_frame, get_x_framev2

import random
import math
import cv2

def _std_normal():
    # Box–Muller：O(1) 生成 N(0,1)
    u1 = 1.0 - random.random()  # 避免 log(0)
    u2 = random.random()
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta)

def make_M_flip_full_single(w_rgb: int) -> np.ndarray:
    return np.array([
        [-1, 0, w_rgb],
        [ 0, 1, 0],
        [ 0, 0, 1]
    ], dtype=np.float32)

import random

import random

def sample_bimodal_edge_gaussian(N, M, K, sigma=3.0, p_left=0.5, exclude_self=True):
    """
    在 [N-K, N+K] 范围内随机采样一个索引（均匀分布）。
    与原版不同：不再具有边缘高斯特性，改为完全随机采样。
    """
    assert 0 <= N < M and M > 0
    if M == 1:
        return 0

    L = max(0, N - K)
    R = min(M - 1, N + K)
    if L == R:
        return L

    # 均匀随机采样
    j = random.randint(L, R)

    # 避免采到自身 N（可选）
    if exclude_self and j == N:
        # 如果只有一个元素，就返回自身
        if L == R:
            return N
        # 否则再重新采一次直到不等于 N
        while j == N:
            j = random.randint(L, R)

    return j




class LasHeR_unregist_trainingSet_framelist(BaseVideoDataset):
    def __init__(self, root=None, dtype='rgbrgb', image_loader=jpeg4py_loader, data_fraction=None, framelist=1, pre_align=False):
        self.root = env_settings().LasHeR_unregist_dir if root is None else root
        super().__init__('LasHeR_unregist_trainingSet_framelist', root, image_loader)

        self.sequence_list = ['rightgirlplayingphone', 'skirtwoman', 'higherthwartbottle_quezhen', 'boybesidesblkcarrunning', 'blkboy', 'pingpongpad', 'twoperpson_1202', 'righttallnine-gai', 'rightshiningmirror', 'boyshorts', 'takeoutman953', 'blkboywithwhitebackpack', 'rightboyatwindow', 'oldwoman', 'meituanbike', 'boywithshorts', 'two_1227', 'bikecoming', 'leftredflag-lsz', '1handsth', 'motostraught2east', 'girlturnbike', 'bikeout', 'tworightbehindboy-gai', 'boydown', 'guardatbike_ab', 'motosmall', 'rightgreenboy', 'blkmoto', 'blkcar2north', '3rdboy', 'rightboy_1227', '7rightredboy', 'womanstartbike', 'leftmen-chong1', 'ajiandan_boyleft2right', 'girlsquattingbesidesleftbar', 'smallmoto', 'whiteboy1_quezhen', '1strowright2ndgirl', '7two', 'redgirlsits', 'redumbrellagirlcome', 'boyrightthelightbrown', 'whitecarleave198', 'blkumbrella', 'leftlastboy-sq', '2whitegirl', 'womanaroundcar', 'rainywhitecar', 'rightwhiteboy', 'whitegirl_0115', '1rowleft2ndgirl', 'redmotocome', 'girlshead', 'takeoutmototurn', 'cameraman_1202', 'fallenbikeitself', 'moto2north101', '6walkgirl', 'elector_1227', 'boy_0109', 'girltakemoto', 'lightcarcome', 'whitecatjump', 'left_two_0109', 'mototurnleft', 'whitewoman_1202', 'umbellaatnight', 'firstrightflagcoming', 'tallwhiteboy', 'trimototurn', 'carstart2east', 'rightboywithbluebackpack', '4one', 'boywithumbrella', 'blkteacher`shead', 'boytakingcamera', 'midboyplayingphone', 'blkgirlfat_quezhen', 'dogunderthelamp', '2gointrees', 'blkboyonleft', 'leftbroom', 'blkcarcome155', 'moto2west', 'mototakinggirl', 'girltakingblkumbrella', '5one', 'girlsheadwithhat', 'righthand`sfoam', 'girlwithumbrella', 'ajiandan_blkdog', '5runtwo', 'ab_bikeboycoming', 'whiteboyright', 'boyrightrubbish', 'nikeatbike', 'girlafterglassdoor2', 'rightblkboybesidesred', 'rightboywitjbag', 'rightholdball', 'right2ndblkboy', 'leftconergirl', '8lastone', '2runone', 'pinkgirl', 'rightexercisebookwillfly', 'lightcarstop', 'manfromcar302', 'manleftsmallwhitehouse', 'whitegirl2_0115', 'besom4', 'yellowtruck', 'blktakeoutmoto', 'tallboyNumber_9', '2boyscome245', 'trolleywith2boxes1_quezhen', 'boywithwhitebackpack', 'boyinsnowfield2', 'pingpongball', 'ab_mototurn', 'lastof4boys', 'motoprecede', 'boywithshorts2', 'midboyblue', 'gonemoto_ab', 'bikewithbag', 'rightbhindbike-gai', 'motowithblack', '9whitegirl', 'ab_hyalinepaperatground', 'blueboycome', 'left3boycoming', 'leftaloneboy-gai', 'midblkbike', 'manwalkincars', 'openningumbrella', 'leftconerbattle-gai', '10rightboy', 'blkboybike', 'theleftboytakingball', 'Take_an_umbrella_1202', 'hatboy`shead', 'moto2trees', 'manfarbesidespool', 'openthisexersicebook', 'blkboywithbluebag', 'bikecome', 'boysumbrella3', 'whitecarinrain', 'rightblkgirl', 'girlatwindow', 'blackdownball', 'whitecarturn137', 'greyboysit1_quezhen', 'girlshakeinrain', '5runone', 'the4thboystandby', 'bikeboycome', 'boyfromdark2', 'whiteboyup', 'bluelittletruck', 'whitebikebehind172', 'blueboybike', 'carfromnorth', 'lowerfoam2throw', 'blkboywillstand', 'boyalonecoming', 'orangegirl', '2runfive', 'singleboywalking', 'rightgirlbikecome', 'farmanrightwhitesmallhouse', 'bluemanatbike', 'boybehindtrees2', 'stubesideswhitecar', 'ab_motoinrain', 'whitacatfrombush', 'bolster', 'wanderingly_1202', 'right2nddrillmaster', 'manbesideslight', 'girltakingmoto', 'motocomeinrain', 'agirl_1202', 'rightbottle2-gai', "comingboy'shead", 'large3-gai', 'rightgirlatbike', 'The_girl_with_the_cup_1202', 'boyatbluegirlleft', 'leftboy-gai', 'dotat43', 'Amidredgirl', 'bowblkboy1-quezhen', 'car2north3', '1righttwogreen', 'ajiandan_catwhite', 'left2flagfornews', 'boytoleft_inf_white', 'rightblkboy188', 'righttallholdball', 'girloutreading', 'rightblkgirlNo_11', 'motocometurn', 'rightmirrorlikesky', '1strowleft3rdgirl', 'leftdrillmasterstanding', 'twopeople_0109', 'whitecarcome192', 'leftblkboyunderbasketballhoop', 'moto2north', 'rightgirl', 'downwhite_1227', 'rightof2boys', 'girlwithredhat', 'whitecarstart126', 'rightwhitegirl', 'boysumbrella2', '9handlowboy', 'pingpongball2', 'greenrightblack', 'mototaking2boys', 'bolster_infwhite', '10phone_boy', 'leftlastgirl2', 'darkgiratbike', 'whitecarcoming', 'left11', 'boyplayingphone', 'moto2north1', 'whitecarturn85', 'AQtaxi', 'leftrunningboy', 'yellowumbrellagirl', 'highright2ndboy', 'bluetruck', '7runtwo', 'takeoutmoto521', 'rightredboy954', '2ndboyfarintheforest2right', 'boy', 'bluegirl', 'rightdrillmasterunderthebar', 'boyunderleftbar', '7runthree', 'carlight', 'girlplayingphone', 'boyputtinghandup', '1boygo', 'whiteminibus197', 'farwhiteboy', 'peoplefromright_0109', 'outerfoam', 'motolight', 'boy_1227', 'motoinrain56', 'manafetrtrees', 'bikefromnorth257', 'besom-ymm', 'blkcargo', 'manupstairs', 'whitegirl1227', 'comebike', 'twolinefirstone-gai', '2ndbikecoming', 'greenboy438',  'blueumbrellagirl', 'maninfrontofbus', 'man2startmoto', 'whitemancome', 'left2ndgreenboy', 'right4thboy', 'leftboyleftblkbackpack', 'girl`sheadoncall', 'man_0109', 'rightboyleader', 'fatmancome', 'bouheadupstream', 'boywithblkbackpack', 'redgirl2trees', 'nearmangotoD', 'whitecat', 'rightbottle-gai', 'car2north', 'strongboy`head', 'boy2_1227', 'boyride2trees', 'rightbottle', 'midwhitegirl', 'runningwhiteboy', 'right2ndfarboytakinglight2left', 'bluemoto', 'pinkgirl285', '2boysatblkcarend', 'meituanbike2', 'nearstrongboy', 'boywalkinginsnow3', '2ndgirlmove', 'boywalkinginsnow', 'girlatleft', 'the2ndboyunderbasket', 'rightumbrella_quezhen', 'boybackpack', 'basketballup', 'ab_leftfoam', 'whitecarstart183', 'boycoming', 'leftbottle', 'left3rdgirlbesideswhitepants', 'lastblkboy1_quezhen', 'The_one_on_the_left_in_black_1202', 'agirl1_1202', 'Agirlrideback', 'rightfirstgirl-gai', 'schoolofeconomics-yxb', 'elegirl', 'whitecarafterbike', 'aboyleft_1202', 'moto2', 'Aboydownbike', 'left4throwboy', 'dogouttrees', '7rightwhitegirl', 'firstleftrunning', '4runone', '1whiteteacher', 'rightrunninglatterone', 'rightboywithwhite', 'boybikewithbag', 'blkcarstart', 'girlintrees', 'leftcup', 'exercisebook', 'minibus152', 'midtallboycoming', '11runone', 'ab_pingpongball3', '2ndrunningboy', 'abeauty_1202', 'e-tricycle', 'lastgirl-qzc', 'whitesuvstop', 'boy_0115', 'bikefromnorth2', '5runfour', '7runone', 'bikeumbrellacome', 'lefthyalinepaper', '7rightblueboy', 'bikeboystrong', 'ab_pingpongball', 'lightmotocoming', 'whitegirl2', 'bluegirlriding', 'easy_whiterignt2left', 'blkboyatbike', 'right2ndblkpantsboy', 'camera_1202', 'blkridesbike', 'leftgirlunderthelamp', '1strowright1stboy', 'rightboy479', '11righttwoboy', 'leftmirrorshining', 'actor_1202', 'doginrain', 'right1stgirlin2ndqueue', 'bikeafterwhitecar', 'boybesidescarwithouthat', 'greenfaceback', 'rightboybesidesredcar', 'girlrightcomein', 'whiteof2boys', 'midblkboyplayingphone', 'lefthandfoamboard', 'lightredboy', 'motocominginlight', '2ndcarcome', 'bookatfloor', 'dogforward', 'bikeblkturn', 'motofromdark', 'sisterswithbags', 'blackthree_1227', 'leftgirlafterlamppost', '2runsix', 'rightwhite_1227', 'boybesidesbar2putcup', 'biketurnleft2', 'shotmaker2', 'blkcarinrain', '4runeight', 'redgirlafterwhitecar', '11runthree', 'left4thgirlwithwhitepants', 'frontmirror', 'shunfengtribike', 'carleaves', 'redlittleboy', 'manaftercars', '2outdark', '10crosswhite', 'ab_minibusstops', 'leftbrowngirlfat', 'blkmotocome', 'bikecoming176', 'bikefromwest', 'yellowexcesicebook', 'firstboythroughtrees', 'whiteboyatbike', 'blkboyback636', 'redgirl', 'outer2leftmirrorback', 'boybetween2blkcar', 'car', 'stronggirl', 'rightblkboy2', 'girlbikeinlight', 'rightestblkboy2', '4three', 'darkleftboy2left', 'boyrideoutandin', 'rainysuitcase', 'midflag-qzc', 'runninggreenboyafterwhite', 'girlruns2right', 'rightgreen', 'easy_rightblkboywithgirl', 'blackof4bikes', 'basketballatboysrighthand', '2runfour', 'blkboycoming', 'take-out-motocoming', 'AQmotomove', 'girl', 'whitegirlundertheumbrella', 'blkmototurn', 'e-tribike', 'blackboypushbike', 'boyridesbike', 'twopeopleelec-gai', 'boytakesuicase', 'boymototakesgirl', 'greenboy', 'right2ndgirl', '4four', 'rightblkgirlrunning', 'collegeofmaterial-gai', 'leftorangeboy', 'leftblkboy', 'fardarkboyleftthe1stgirl', 'whitegirl2right', 'rightof2cominggirls', 'boyshead509', 'left2ndboy', 'mototurntous', 'bike2north', 'motocome2left', 'boycomingwithumbrella', 'Ahercarstart', 'yellowcar', 'girlback', 'leftwhitebike', 'rightboystand', 'the4thwhiteboy', '7one', 'blackcarturn183', 'nightboy', 'waiterontheothersideofwindow', 'blkcarinrain107', 'leftlightsweaterboy', 'rightbiggreenboy', 'leftgirl1299', '4boysbesidesblkcar', 'boyplayingphone366', 'ab_leftmirrordancing', 'bus2north', 'bikeblkbag', 'ab_righthandfoamboard', 'carfromnorth2', 'blkbagontheleftgirl', 'boybikeblueumbrella', 'bluegirlcoming', 'blackcarcoming', 'girlinfrontofcars', 'rightmirrornotshining', '4two', 'left4thblkboyback', 'whiteboyback', 'blklittlebag', 'rightholdball1096', 'glassesboyhead', 'rainblackcarcome', 'raincarstop2', 'orangegirlwithumbrella', 'leftshortgirl', 'lastrowrightboy', 'leftgirl', 'left2ndgirl', 'leftboy', 'The_girl_back_at_the_lab_1202', 'whitemoto', 'leftgirlat1row', 'yellowgirlwithbowl', '1rightboy', 'boyinsnowfield4', 'rightstripeblack', 'blackcar131', '2girlsridebikes', 'whitebikebehind', 'blkgirlfromcolumn1_quezhen', '2runtwo', 'girloutqueuewithbackpack', 'girltakingplate', 'redbaggirlleftgreenbar', 'bike2trees86', 'blkskirtwoman', 'blkgirlbike', 'midgirl', 'girlbesidesboy', 'boybehindtrees', 'ab_catescapes', 'rightboy`shead', 'lover_1202', 'blkboy198', 'blkcarcome115', 'belowyellow-gai', 'rainyboyaftertrees', 'mototake2boys123', 'rightofth3boys', 'rightblkboy', 'browncar2north', 'blackdresswithwhitefar', 'rightofthe4girls', 'easy_runninggirls', 'greenleftthewhite', 'large2-gai', 'leftlastgirl-yxb', 'boyindarkwithgirl', 'whitecar2west', 'runningwhiteboy249', 'fogboyscoming1_quezhen_inf_heiying', 'girlridesbike', 'whiteboy242', 'biketurnleft', 'AQmidof3boys', 'camonflageatbike', 'whitemotoout', 'blackbetweengreenandorange', 'rightgirlbike', 'manrun', 'lovers_1227', 'twoleft', 'bike2', 'nearestleftblack', 'leftgirlchecking', 'raincarstop', '4thgirl', 'ab_moto2north0', 'leftmirror2', 'carcomeonlight2', 'blackruning', 'blackcarback', 'leftbasketball', '5manwakeright', 'truckcoming', 'girlwithblkbag', 'whitecarturnright248', 'motocoming', 'yellowatright', 'foamboardatlefthand', 'whitehatgirl`sheadleftwhiteumbrella', 'whitegirl', 'battlerightblack', 'whiteboybike', 'mototurn134', 'leftclosedexersicebook', 'whitegirlatbike', 'blkcarcome', 'manatmoto', 'rightbattle', '9hatboy', 'rightgirlwithumbrella', 'carturncome', 'minibusback', 'motocome', 'blkmaninrain', 'ab_rightcupcoming_infwhite_quezhen', 'motowithtop', 'whitecarfromnorth', 'lightcarfromnorth', 'belowrightwhiteboy', 'blkboywithblkbag', 'girl2-gai', 'besom5-sq', 'car2north2', 'besom6', 'blkboygoleft', 'midpinkblkglasscup', 'carturnleft109', 'catbrown', 'boycome', 'lonelyman', 'blackboy186', 'manwithyellowumbrella', 'orange', 'browncar2east', 'folderatlefthand', 'farwhitecarturn', 'folddenumbrellainhand', 'blkcarcomeinrain', 'biketurn', 'redgirl1497', 'guardman', 'redboywithblkumbrella', 'basketballbyNo_9boyplaying', 'boyaftertworunboys', 'greenleftbackblack', 'redcupatleft', 'small2-gai', 'rightbackcup', '1phoneblue', 'ab_redboyatbike', 'pickuptheyellowbook', 'leftblkboy648', '2boysbesidesblkcar', 'boyshead', 'blackbagbike', '1strow3rdboymid', 'bord', 'blkboywithumbrella', 'turnblkbike', 'blkman2trees', 'boyfollowing', 'blueboy85', 'basketballatright', '2ndblkboy1_quezhen', 'leftfallenchair_inf_white', 'boyblackback', 'leftdrillmasterstandsundertree', 'motoinrain', 'thefirstexcersicebook', 'blkbikefromnorth', 'left3rdrunwaygirlbesideswhitepants', 'rightof2cupsattached', 'rightredcup_quezhen', 'redbaginbike', 'leftwhiteblack', '5runthree', 'boyof2leaders', 'rightestblkboy', 'whitegirlcoming', 'leftthrowfoam', 'twoperson_1202', 'righthunchblack', 'rightfirstboy-ly', 'blkcarfollowingwhite', 'blueboyopenbike', 'carstarts', 'othersideoftheriver_1227', 'bluebike', 'girlthroughtrees', 'manatwhiteright', 'blackman_0115', 'boy1227', 'manfromcar', 'greenboyafterwhite', 'bikefromnorth', 'boystandinglefttree', 'aboy_1202', 'notebook-gai', 'redminirtruck', 'left_first_0109', 'leftof2girls', 'redbackpackgirl', 'folderinrighthand', 'rightwhitegirlleftpink', 'ninboy-gai', 'theleftestrunningboy', '10rightblackboy', 'toulan-ly', 'boyunderthecolumn', 'boyshead2', 'midboy', 'boyinsnowfield_inf_white', 'boyouttrees', 'blueboywalking', 'sitleftboy', 'rightredboy1227', 'motolightturnright', 'suitcase', 'swan2_0109', 'boyleft', 'backpackboyhead', 'blkboylefttheredbagboy', 'motowithtopcoming', 'rightof2boys953', 'manbikecoming', 'ab_rightmirror', 'basketballshooting', 'moto2north2', 'besom2-gai', 'boysumbrella', 'pinkbikeboy', 'rightmirrorbackwards', 'the4thboy', 'whitecarback', 'whitegirlcrossingroad', 'basketballshooting2', 'blkbikecomes', 'abreastinnerboy', 'dogfollowinggirl', 'whiteboy-gai', 'ab_bolster', 'leftdress-gai', 'whiteboy`head', 'hyalinepaperfrontclothes', 'whiteshoesleftbottle-gai', 'boy2_0115', 'mantoground', 'elector_0115', 'bikeorange', 'mirroratleft', 'boyrunning', 'moto2trees2', 'rightredboy', 'whitebetweenblackandblue', 'nightrightboy1', 'foundsecondpeople_0109', 'leftredcup', 'whitesuvcome', 'mirrorfront', 'rightboywithbackpackandumbrella', 'bikeboy128', 'whitegirlwithumbrella', '2ndbus', 'whitecarcomes', '1strowrightgirl', 'boyridesbesidesgirl', 'left_leader', 'boyumbrella4', 'biketonorth', 'blkboywithglasses', 'darkredcarturn', 'pingpongpad2', 'drillmaster']

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        self.framelist = framelist
        self.margin = 300
        self.dtype = dtype
        self.pre_align = pre_align
    def get_name(self):
        return 'LasHeR_unregist_trainingSet_framelist'

    def _read_bb_anno_v(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'visible.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)
    
    def _read_bb_anno_i(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'infrared.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        bbox_v = self._read_bb_anno_v(seq_path)
        bbox_i = self._read_bb_anno_i(seq_path)
        valid = (bbox_v[:, 2] < 10000) & (bbox_v[:, 2] > 10) & \
                (bbox_v[:, 3] < 10000) & (bbox_v[:, 3] > 10) & \
                (bbox_v[:, 0] < 10000) & (bbox_v[:, 0] > 0)  & \
                (bbox_v[:, 1] < 10000) & (bbox_v[:, 1] > 0)  & \
                (bbox_i[:, 2] < 10000) & (bbox_i[:, 2] > 10) & \
                (bbox_i[:, 3] < 10000) & (bbox_i[:, 3] > 10) & \
                (bbox_i[:, 0] < 10000) & (bbox_i[:, 0] > 0)  & \
                (bbox_i[:, 1] < 10000) & (bbox_i[:, 1] > 0)
            
        visible = valid.clone().byte()
        return {'bbox_rgb': bbox_v, 'bbox_ir': bbox_i, 'valid': valid, 'visible': visible}
    
        # return {'bbox_v': bbox_v, 'bbox_i': bbox_i, 'valid': valid, 'visible': visible}
        
        
    def get_num_sequences(self):
        return len(self.sequence_list)
    
    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'visible', sorted([p for p in os.listdir(os.path.join(seq_path, 'visible')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return frame_path_v
        # return self.image_loader(frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'infrared', sorted([p for p in os.listdir(os.path.join(seq_path, 'infrared')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return frame_path_i
        # return self.image_loader(frame_path_i)
    
    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path = self._get_frame_v(seq_path, frame_id)
        ir_frame_path = self._get_frame_i(seq_path, frame_id)
        
        img_rgb_size,img_tir_size = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        
        return img_rgb_size, img_tir_size

 
    def _align_images(self, tir_image, rgb_bbox, tir_bbox, warped_tir_bbox, h_rgb, w_rgb, h_tir, w_tir, pre_aligned=True):
        return self._align_numpy(tir_image, rgb_bbox, tir_bbox, warped_tir_bbox, h_rgb, w_rgb, h_tir, w_tir, pre_aligned)

    def _align_numpy(self, tir_image, rgb_bbox, tir_bbox, warped_tir_bbox, h_rgb, w_rgb, h_tir, w_tir, pre_aligned):
        """
        使用单应性矩阵对齐 TIR 图像到 RGB 图像

        rgb_image: HxWxC, numpy 或 tensor
        tir_image: HxWxC, numpy 或 tensor
        rgb_bbox: [x, y, w, h] in RGB
        tir_bbox: [x, y, w, h] in TIR
        """
        
        # 确保整数
        rgb_bbox = np.asarray(rgb_bbox, dtype=np.float32)
        tir_bbox = np.asarray(tir_bbox, dtype=np.float32)

        # 按同样比例缩放 TIR bbox
        scale_x = w_rgb / w_tir
        scale_y = h_rgb / h_tir
        tir_bbox_scaled = np.array([
            tir_bbox[0]*scale_x, tir_bbox[1]*scale_y,
            tir_bbox[2]*scale_x, tir_bbox[3]*scale_y
        ], dtype=np.float32)
        
        warped_tir_bbox_scaled = np.array([
            warped_tir_bbox[0]*scale_x, warped_tir_bbox[1]*scale_y,
            warped_tir_bbox[2]*scale_x, warped_tir_bbox[3]*scale_y
        ], dtype=np.float32)
        
        if not pre_aligned: 
            return tir_image, warped_tir_bbox_scaled
        
        
        # 定义四个角点 (x, y)
        def bbox_to_corners(bbox):
            x, y, w, h = bbox
            return np.array([
                [x, y],           # 左上
                [x + w, y],       # 右上
                [x + w, y + h],   # 右下
                [x, y + h]        # 左下
            ], dtype=np.float32)
        
        pts_rgb = bbox_to_corners(rgb_bbox)
        pts_tir = bbox_to_corners(tir_bbox_scaled)
        
        # 计算单应性矩阵
        H, _ = cv2.findHomography(pts_tir, pts_rgb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        
        aligned = cv2.warpPerspective(
            tir_image, H, (w_rgb, h_rgb),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        warped_bbox = None
        if warped_tir_bbox_scaled is not None:
            pts_warp = bbox_to_corners(warped_tir_bbox_scaled).reshape(-1, 1, 2)
            pts_warped = cv2.perspectiveTransform(pts_warp, H).reshape(-1, 2)
            # 重新取最小外接矩形
            x_min, y_min = pts_warped[:,0].min(), pts_warped[:,1].min()
            x_max, y_max = pts_warped[:,0].max(), pts_warped[:,1].max()
            warped_bbox = np.array([x_min, y_min, x_max - x_min, y_max - y_min], dtype=np.float32)
            
        return aligned, warped_bbox

    def _get_bias_bbox_ir(self, bbox_rgb, dx, dy):
        bbox_ir_bias = bbox_rgb + torch.tensor([dx, dy, 0, 0])
        return bbox_ir_bias
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        frame_length = len(os.listdir(os.path.join(seq_path, "visible")))
            
        frame_list_rgb_size = []
        frame_list_tir_size = []
        frame_list_rgb_prior = []
        frame_list_tir_prior = []
        sample_frameids = []
        for f_id in frame_ids:
            frame_rgb, frame_tir = self._get_frame(seq_path, f_id)
            h_rgb, w_rgb = frame_rgb.shape[:2]
            h_tir, w_tir = frame_tir.shape[:2]
            sampled_fid = sample_bimodal_edge_gaussian(f_id, frame_length, self.margin)
            frame_tir_weakly_align, weakly_align_bbox = self._align_images(frame_rgb[:,:,3:], anno['bbox_rgb'][sampled_fid],
                                                            anno['bbox_ir'][sampled_fid] , anno['bbox_ir'][f_id], h_rgb, w_rgb, h_tir, w_tir, pre_aligned=self.pre_align)
            
            frame_rgb = np.concatenate([frame_rgb[:,:,:3], frame_tir_weakly_align], axis=2)
            frame_list_rgb_size.append(frame_rgb)
            frame_list_tir_size.append(frame_tir)
            sample_frameids.append(sampled_fid)
            anno_ir = torch.from_numpy(weakly_align_bbox)
            
        
        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        anno_frames_prior = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            anno_frames_prior[key] = [value[max(0, f_id-1), ...].clone() for f_id in frame_ids]
            
                    
        anno_frames['bbox_ir_bias'] = [self._get_bias_bbox_ir(anno_ir, 0, 0)]
        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        
        return frame_list_rgb_size, frame_list_tir_size, frame_list_rgb_prior, frame_list_tir_prior, anno_frames, anno_frames_prior, object_meta, seq_name, sampled_fid, h_rgb, w_rgb
