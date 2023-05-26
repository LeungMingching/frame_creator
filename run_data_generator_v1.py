import os
import json
import random
import time
import copy
import numbers
from glob import glob
from tqdm import tqdm
import numpy as np
from utils import *
from matplotlib import pyplot as plt


def arc_to_theta(arc_length, radius):
    return arc_length/radius

# 设置一系列曲率不同的圆,从各圆上获取不同曲率的曲线
# 最小转弯半径为6m,则最大曲率假定为1/R_min=1/6
#rho_list = [1/10.0, 1/25.0, 1/55.0, 1/100.0, 1/130.0, 1/300.0, 1/500.0, 1/1000.0]
#rho_list = [1/55.0, 1/130.0, 1/300.0, 1/1000.0]
rho_list = [1/1000.0]

# 3车道(0，1，2号)，ego会出现0号，1号，2号上，发起左换道，右换道，车道保持等
# 仅假设v_ego为36km/h，换道粗略按照5s内完成换道的策略
# VTD下车道线通常最大给100m

# 每条曲线上采集的最大样本点数目
max_num_sample_pts = 200

# 使用tanh生成近似的换道轨迹
sin_y_bound = 1.0
#d_lane = [3.0, 3.3, 3.5, 3.7]
d_lane = [3.5]

# 将目标参考线分成若干份,比如25份
num_blks_on_target_ref_line = 25

def rotate(path, theta):
    rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]], dtype=np.float16)
    return np.matmul(path, rotate_mat)

def transfer(path, vec):
    return path + vec

def generate_planned_path(s, lateral_bias, trend='lk'):
    '''
    输入
        s:  路径规划所覆盖参考线上的纵向距离
        lateral_bias: 路径规划所覆盖的横向距离
        trend: 路径规划的机动类型, 'lk'-车道保持，'lc'-换道
    '''
    assert trend in ['lk','lc'], "Invalid trend type!"

    if trend=='lk':
        s = 3.0 if s<3.0 else s
    else:
        s = 5.0 if s<5.0 else s
        

    # 对s进行整形以保证s是0.5的倍数
    num_points = int(np.ceil(s/0.5)) + 1
    s = (num_points-1)*0.5
    s_samples = np.linspace(0.0, s, num_points).astype(np.float16)

    # lateral_bias遵循左正右负原则
    if trend == 'lk':
        d_samples = (-np.sin(np.pi*s_samples/s-0.5*np.pi) + sin_y_bound) * (-0.5*lateral_bias)
    else:
        d_samples = (np.sin(np.pi*s_samples/s-0.5*np.pi) + sin_y_bound) * (-0.5*lateral_bias)

    d_samples = d_samples.astype(np.float16)
    planned_path = np.hstack((s_samples[:,np.newaxis], d_samples[:,np.newaxis]))
    planned_path = rotate(planned_path, -0.5*np.pi) # 旋转后正好是左正上正, 契合车辆坐标系的朝向

    return planned_path.astype(np.float16)


def to_bev(path, ego_pos, ego_heading):
    vec = -ego_pos
    # 1.平移
    path = transfer(path, vec)
    # 2.旋转
    if ego_heading!=0.0:
        path = rotate(path, -ego_heading)

    return path.astype(np.float16)


def heading_angle_bound_by_velocity(vel):
    assert vel>=0.0 and vel<=20.0, "速度超出0~20m/s!"

    angle_high = np.pi/9
    angle_low = np.pi/36
    vel_range = 20.0

    return (angle_low-angle_high)/vel_range*vel + angle_high

'''
right_planned_path = generate_planned_path(10.0, -3.5, trend='lc')
right_planned_path = transfer(right_planned_path, np.array([0.0,0.0]))
# 车辆坐标系下，逆时针旋转角度为+，顺时针旋转角度为-，旋转坐标系则相反
# 在正常x-y坐标系下,顺时针旋转角度为+,逆时针为-，与车辆坐标系下的情况相反
#right_planned_path = to_bev(right_planned_path, np.array([0.0, 0.0]), -0.25*np.pi) # 这里旋转角度在车辆坐标系下
fig = plt.figure()
plt.axis('equal')
plt.plot(right_planned_path[:,0], right_planned_path[:,1])
#plt.scatter(left_planned_path[:,0], left_planned_path[:,1], s=5, c='r')
#plt.scatter(right_planned_path[:,0], right_planned_path[:,1], s=5, c='b')
plt.show()
exit(0)
'''

def cmp(a, b):
    if a>b:
        return 1
    elif a<b:
        return 2
    else:
        return 0


# 数据集
# input
Reference_Line_Cluster_List_Train = []
Reference_Line_Mask_Train = []
Planned_Path_List_Train = []
# None->0, ToLeft->1, ToRight->2
Navi_Drive_List_Train = []
# v_ego, (heading, location)
Ego_State_List_Train = []
# lk->0, llc->1, rlc->2
Current_Context_List_Train = []
# prediction
# lk->0, llc->1, rlc->2
Next_Context_List_Train = []
# x, y, v
Lookahead_Info_Cluster_List_Train = []
# lateral distance to ego, target reference line
Lateral_Dist_To_Ego_Ref_Line_Train = []
Lateral_Dist_To_Target_Ref_Line_Train = []
# 正数, 负数编码(0-非负数, 1->负数)
Ego_Bias_Pos_Neg_Info_Train = []
Target_Bias_Pos_Neg_Info_Train = []
# (0--=, 1-->, 2--<)
Abs_Bias_Compare_Info_Train = []
# 目标参考线的分布，比如00010
Target_Reference_Line_Distrib_Train = []
# ego, target参考线分布, 比如00110
Ego_Target_Reference_Line_Distrib_Train = []
# frenet s of target reference line
Frenet_S_On_Reference_Line_Train = []

# input
Reference_Line_Cluster_List_Test = []
Reference_Line_Mask_Test = []
Planned_Path_List_Test = []
# None->0, ToLeft->1, ToRight->2
Navi_Drive_List_Test = []
# v_ego, (heading, location)
Ego_State_List_Test = []
# lk->0, llc->1, rlc->2
Current_Context_List_Test = []
# prediction
# lk->0, llc->1, rlc->2
Next_Context_List_Test = []
# x, y, v
Lookahead_Info_Cluster_List_Test = []
# lateral distance to ego, target reference line
Lateral_Dist_To_Ego_Ref_Line_Test = []
Lateral_Dist_To_Target_Ref_Line_Test = []
# 正数, 负数编码(0-非负数, 1->负数)
Ego_Bias_Pos_Neg_Info_Test = []
Target_Bias_Pos_Neg_Info_Test = []
# (0--=, 1-->, 2--<)
Abs_Bias_Compare_Info_Test = []
# 目标参考线的分布，比如00010
Target_Reference_Line_Distrib_Test = []
# ego, target参考线分布, 比如00110
Ego_Target_Reference_Line_Distrib_Test = []
# frenet s
Frenet_S_On_Reference_Line_Test = []


def data_generator(Reference_Line_Cluster_List_Train,
                   Reference_Line_Mask_Train,
                   Planned_Path_List_Train,
                   Navi_Drive_List_Train,
                   Ego_State_List_Train,
                   Current_Context_List_Train,
                   Next_Context_List_Train,
                   Lookahead_Info_Cluster_List_Train,
                   Lateral_Dist_To_Ego_Ref_Line_Train,
                   Lateral_Dist_To_Target_Ref_Line_Train,
                   Ego_Bias_Pos_Neg_Info_Train,
                   Target_Bias_Pos_Neg_Info_Train,
                   Abs_Bias_Compare_Info_Train,
                   Target_Reference_Line_Distrib_Train,
                   Ego_Target_Reference_Line_Distrib_Train,
                   Frenet_S_On_Reference_Line_Train,
                   Reference_Line_Cluster_List_Test,
                   Reference_Line_Mask_Test,
                   Planned_Path_List_Test,
                   Navi_Drive_List_Test,
                   Ego_State_List_Test,
                   Current_Context_List_Test,
                   Next_Context_List_Test,
                   Lookahead_Info_Cluster_List_Test,
                   Lateral_Dist_To_Ego_Ref_Line_Test,
                   Lateral_Dist_To_Target_Ref_Line_Test,
                   Ego_Bias_Pos_Neg_Info_Test,
                   Target_Bias_Pos_Neg_Info_Test,
                   Abs_Bias_Compare_Info_Test,
                   Target_Reference_Line_Distrib_Test,
                   Ego_Target_Reference_Line_Distrib_Test,
                   Frenet_S_On_Reference_Line_Test,
                   test_split=0.01):

    # 以ego视角为基准，最多考虑5条车道，与ego最近的1条作为ego(并非ego lane), ego两侧各排列2条
    # XXX: 定义一个以ego参考线为中心的局部地图坐标系，最多5条参考线路点均给出该坐标系下的坐标
    # 若5条参考线(排序为3-1-0-2-4，0对应0号参考线)均为非空，则称之为valid，否则称之为invalid
    valid_ref_line_0 = []
    valid_ref_line_1 = []
    valid_ref_line_2 = []
    valid_ref_line_3 = []
    valid_ref_line_4 = []
    invalid_ref_line = []

    # 假设路点间隔为0.5米
    lane_width = 3.5
    num_wps_behind = 40
    total_num_waypoints = 400 + num_wps_behind
    for i in range(total_num_waypoints):
        # 左正右负
        x0, y0 = (i-num_wps_behind)*0.5, 0.0
        x1, y1 = (i-num_wps_behind)*0.5, 1.0*lane_width
        x3, y3 = (i-num_wps_behind)*0.5, 2.0*lane_width
        x2, y2 = (i-num_wps_behind)*0.5, -1.0*lane_width
        x4, y4 = (i-num_wps_behind)*0.5, -2.0*lane_width
        valid_ref_line_0.append(np.array([y0, x0], dtype=np.float16))
        valid_ref_line_1.append(np.array([y1, x1], dtype=np.float16))
        valid_ref_line_2.append(np.array([y2, x2], dtype=np.float16))
        valid_ref_line_3.append(np.array([y3, x3], dtype=np.float16))
        valid_ref_line_4.append(np.array([y4, x4], dtype=np.float16))
        invalid_ref_line.append(np.array([0.0, 0.0], dtype=np.float16))

    # 注意顺序是3-1-0-2-4
    valid_ref_lines = [
                        np.array(valid_ref_line_3), 
                        np.array(valid_ref_line_1), 
                        np.array(valid_ref_line_0), 
                        np.array(valid_ref_line_2), 
                        np.array(valid_ref_line_4)
                      ]
    invalid_ref_line = np.array(invalid_ref_line)

    ####################################################################
    # 一共有9中可能性
    # 5条车道的有效性描述(0-无效，1有效)
    # ==================================================================
    # 车道序号                  3   1   0   2   4
    # ----------------------------------+-------------------------------
    # case 0:                   0   0   1   0   0   ----->  (lk)
    # ----------------------------------+-------------------------------
    # case 1:                   0   0   1   1   0   ----->  (lk, rlc)
    # ----------------------------------+-------------------------------
    # case 2:                   0   1   1   0   0   ----->  (lk, llc)
    # ----------------------------------+-------------------------------
    # case 3:                   0   1   1   1   0   ----->  (lk, llc, rlc)
    # ----------------------------------+-------------------------------
    # case 4:                   0   0   1   1   1   ----->  (lk, rlc)
    # ----------------------------------+-------------------------------
    # case 5:                   1   1   1   0   0   ----->  (lk, llc)
    # ----------------------------------+-------------------------------
    # case 6:                   0   1   1   1   1   ----->  (lk, llc, rlc)
    # ----------------------------------+-------------------------------
    # case 7:                   1   1   1   1   0   ----->  (lk, llc, rlc)
    # ----------------------------------+-------------------------------
    # case 8:                   1   1   1   1   1   ----->  (lk, llc, rlc)
    # ----------------------------------+-------------------------------
    #
    # navi drive: 0-导航不触发换道，1-导航触发向左换道，2-导航触发向右换道
    
    actions = {'lk': 0, 'llc': 1, 'rlc': 2}
    context_types = {'lane_keep': 0, 'left_lane_change': 1, 'right_lane_change': 2}
    case_info_list = [
                        {'lane_valid': [0, 0, 1, 0, 0], 'can_do': ['lk'],               'navi_drive': [0, 1, 2]},
                        {'lane_valid': [0, 0, 1, 1, 0], 'can_do': ['lk','rlc'],         'navi_drive': [0, 1, 2]},
                        {'lane_valid': [0, 1, 1, 0, 0], 'can_do': ['lk','llc'],         'navi_drive': [0, 1, 2]},
                        {'lane_valid': [0, 1, 1, 1, 0], 'can_do': ['lk','llc','rlc'],   'navi_drive': [0, 1, 2]},
                        {'lane_valid': [0, 0, 1, 1, 1], 'can_do': ['lk','rlc'],         'navi_drive': [0, 1, 2]},
                        {'lane_valid': [1, 1, 1, 0, 0], 'can_do': ['lk','llc'],         'navi_drive': [0, 1, 2]},
                        {'lane_valid': [0, 1, 1, 1, 1], 'can_do': ['lk','llc','rlc'],   'navi_drive': [0, 1, 2]},
                        {'lane_valid': [1, 1, 1, 1, 0], 'can_do': ['lk','llc','rlc'],   'navi_drive': [0, 1, 2]},
                        {'lane_valid': [1, 1, 1, 1, 1], 'can_do': ['lk','llc','rlc'],   'navi_drive': [0, 1, 2]}
                     ]

    # 车道保持调整/换道转移所需的最大耗时
    t_lc_max = 5.0
    t_lk_max = 3.0


    # local func
    def withdraw_all_s_list_and_lookahead_points_needed(planned_path, trend):
        assert trend in ['lk', 'llc', 'rlc'], "Wrong trend type!!!"

        max_num_items = len(planned_path) + 220
        total_lookahead_points, total_s_list_on_planned_path = [], []
        for t in range(max_num_items):
            if t<len(planned_path):
                # lookahead
                x = planned_path[t][1]
                y = planned_path[t][0]
                # frenet s
                if t>0:
                    current_s = total_s_list_on_planned_path[-1] + np.linalg.norm(planned_path[t]-planned_path[t-1])
                else:
                    current_s = 0.0
            else:
                # lookahead
                x = t*0.5
                y = 0.0 if trend=='lk' else (1.0*lane_width if trend=='llc' else -1.0*lane_width)
                # frenet s
                current_s = total_s_list_on_planned_path[-1] + 0.5

            total_s_list_on_planned_path.append(current_s)
            total_lookahead_points.append(np.array([y, x], dtype=np.float16))

        # 转成numpy数组以提升计算速度
        total_s_list_on_planned_path = np.array(total_s_list_on_planned_path, dtype=np.float16)
        total_lookahead_points = np.array(total_lookahead_points, dtype=np.float16)

        return total_s_list_on_planned_path, total_lookahead_points 

    # local func
    def sample_required_num_items(req_num_items, 
                                  current_idx_on_planned_path, 
                                  ego_pos_on_path, 
                                  heading_on_path, 
                                  v_ego, 
                                  total_s_list_on_planned_path, 
                                  total_lookahead_points):

        assert req_num_items<=200, "无效req_num_items!!!"

        sample_index = np.linspace(0, 199, req_num_items).astype(np.int32) + current_idx_on_planned_path
        s_list_on_planned_path = total_s_list_on_planned_path[sample_index] - total_s_list_on_planned_path[current_idx_on_planned_path]
        lookahead_points = total_lookahead_points[sample_index]
        # 将lookahead_points进行旋转操作
        lookahead_points = to_bev(lookahead_points, ego_pos_on_path, heading_on_path)
        # 生成lookahead_info_cluster
        lookahead_info_cluster = np.hstack((lookahead_points, np.ones([len(sample_index),1])*v_ego))

        return s_list_on_planned_path, lookahead_info_cluster


    # 开始处理数据
    for case_info in tqdm(case_info_list):
        lane_valid = case_info['lane_valid']
        can_do = case_info['can_do']

        ######################
        # 初始化不同settings
        ######################
        # 1.准备参考的输入lane信息(唯一)
        lane_cluster_setting = []
        lane_valid_record = []
        for i, lv in enumerate(lane_valid):
            lane_cluster_setting.append(valid_ref_lines[i] if lv==1 else invalid_ref_line)
            lane_valid_record.append(lv)
            
        # 2.准备可能的导航navi drive信息(兼具制造干扰因素的作用)
        navi_drive_settings = case_info['navi_drive']

        # 3.准备可能的自车状态ego state信息
        # 自车速度: 0~20.0m/s(用于planning path)
        v_ego_settings = np.linspace(0.0, 20.0, 21).astype(np.float16)

        # 自车位置可以在原车道内横向范围内-1.5~1.5进行横向偏移量的波动(用于planning path)
        #lateral_bias_settings = np.linspace(-1.5, 1.5, 11).astype(np.float16)
        lateral_bias_settings = np.linspace(-1.5, 1.5, 15).astype(np.float16)
        #if 0.0 not in list(lateral_bias_settings):
        #    lateral_bias_settings = np.concatenate((lateral_bias_settings, np.array([0.0], dtype=np.float16)), axis=0)

        lateral_bias_settings_4llc = np.linspace(-3.5, 1.5, 15).astype(np.float16)
        #if 0.0 not in list(lateral_bias_settings_4llc):
        #    lateral_bias_settings_4llc = np.concatenate((lateral_bias_settings_4llc, np.array([0.0], dtype=np.float16)), axis=0)

        lateral_bias_settings_4rlc = np.linspace(-1.5, 3.5, 15).astype(np.float16)
        #if 0.0 not in list(lateral_bias_settings_4rlc):
        #    lateral_bias_settings_4rlc = np.concatenate((lateral_bias_settings_4rlc, np.array([0.0], dtype=np.float16)), axis=0)

        '''
        # 自车的车头朝向可以在原车道内进行-pi/9~pi/9之间角度波动(仅用于参考planned path的坐标转换),左+右-
        heading_angle_settings = np.linspace(-np.pi/9, np.pi/9, 15).astype(np.float16)
        if 0.0 not in list(heading_angle_settings):
            heading_angle_settings = np.concatenate((heading_angle_settings, np.array([0.0], dtype=np.float16)), axis=0)
        '''

        # 4.准备当前时刻上下文current context信息(lk模式下总是为0，lc模式下取决于是否已经开始换道，已经开始则为1/2，否则为0)
        # 必须根据navi_drive类型以及lane validity等情况确定, 此处还无法确定
        current_context_settings = None

        ## 当到目标车道的横向距离低于该值时可认为换道完成，即lc->lk
        lc_to_lk_threshold = 1.5

        ###############################################
        # 根据上述不同的setting组合生成训练输入输出数据
        ###############################################
        allowed_actions = [actions[k] for k in can_do]
        nonconflict_actions = list(set(allowed_actions)&set(navi_drive_settings))
        conflict_actions = []
        for nds in navi_drive_settings:
            if nds not in nonconflict_actions:
                conflict_actions.append(nds)

        assert len(conflict_actions)+len(nonconflict_actions)==len(navi_drive_settings), "conflict_actions, nonconflict_actions处理出错!"

        ############################################
        # 导航指令与道路有效性不冲突
        # 导航指令与道路有效性相冲突(冲突则默认lk)
        ############################################
        for navi_drive in navi_drive_settings:  # 输入导航指令
            ##############################################
            # 1.导航指令与道路有效性相冲突(冲突则默认lk)
            ##############################################
            if navi_drive in conflict_actions:
                for v_ego in v_ego_settings:    # XXX: 1.自车速度
                    # 自车的车头朝向可以在原车道内进行-theta_x~theta_x之间角度波动(仅用于参考planned path的坐标转换),左+右-
                    heading_angle_bound = heading_angle_bound_by_velocity(v_ego) # 自车速度约束可允许的最大车头朝向
                    heading_angle_step = (2.667/180)*np.pi
                    num_intervals = int(np.ceil(2*heading_angle_bound/heading_angle_step))
                    num_intervals = num_intervals+1 if num_intervals%2==0 else num_intervals
                    heading_angle_settings = np.linspace(-heading_angle_bound, heading_angle_bound, num_intervals).astype(np.float16)
                    if 0.0 not in list(heading_angle_settings):
                        heading_angle_settings = np.concatenate((heading_angle_settings, np.array([0.0], dtype=np.float16)), axis=0)
                    #print(v_ego, heading_angle_bound, num_intervals)

                    ego_state = np.array([v_ego, 0.0], dtype=np.float16)    # 输入自车信息(v, a)
                    # FIXME: 速度为零时等于零
                    s = v_ego*t_lk_max  # lk覆盖的纵向距离

                    for bias in lateral_bias_settings:  # XXX: 2.自车横向位置
                        # 在0号参考线坐标系下,即ref line 0所在的坐标系
                        ego_pos = np.array([bias, 0.0], dtype=np.float16) # -1.5~1.5m之间

                        # XXX: lk模式下planned path坐标已经是基于0号参考线坐标系
                        planned_path = generate_planned_path(s, bias, trend='lk')

                        # 一次性将lookahead_info_cluster和s_list_on_planned_path所涉及的所有项都准备好
                        total_s_list_on_planned_path, total_lookahead_points = withdraw_all_s_list_and_lookahead_points_needed(planned_path, 'lk')
                        
                        for i in range(len(planned_path)):
                            # 这里需要根据planned path上不同位置计算对应的lookahead
                            if i>0:
                                ego_pos_on_path = planned_path[i]
                                vector = planned_path[i] - planned_path[i-1]
                                heading_on_path = np.arctan(vector[0]/vector[1])

                                # 1.根据heading angle对lane_cluster进行旋转
                                lane_cluster_transformed = []
                                for j, lane in enumerate(lane_cluster_setting):
                                    if lane_valid_record[j]==1:
                                        lane_transformed = to_bev(lane, ego_pos_on_path, heading_on_path)  # 输入参考线信息,放心使用,to_bev内部已经处理好了符号
                                    else:
                                        lane_transformed = lane

                                    # 只提取x>0部分(保留200个点)
                                    lane_transformed = lane_transformed[lane_transformed[:,1]>=0.0][0:200]

                                    # 一组5条经过平移旋转后的参考线输入(后续的normalization需要注意x,y是反的)
                                    lane_cluster_transformed.append(lane_transformed)

                                # 2.根据heading angle对planned_path进行旋转
                                planned_path_transformed = to_bev(planned_path, ego_pos_on_path, heading_on_path)

                                # XXX: 均匀采样出25个样本
                                s_list_on_planned_path, lookahead_info_cluster = sample_required_num_items(25,
                                                                                                           i,
                                                                                                           ego_pos_on_path, 
                                                                                                           heading_on_path, 
                                                                                                           v_ego, 
                                                                                                           total_s_list_on_planned_path, 
                                                                                                           total_lookahead_points)

                                # 处理planned path, fix size
                                ppt = planned_path_transformed[planned_path_transformed[:,1]>=0.0]
                                if len(ppt)>=200:
                                    ppt = ppt[0:200]
                                else:
                                    ppt = np.vstack((ppt, np.ones([200-len(ppt),2])*ppt[-1]))

                                # context
                                #current_context = context_types['lane_keep']        # 输入当前上下文
                                # 尽可能增加current_context噪声使得对next context的预测鲁棒性更高
                                for _, context in context_types.items():
                                    current_context = context

                                    next_context = context_types['lane_keep']

                                    # 输入输出数据集
                                    Reference_Line_Cluster_List_Train.append(np.array(lane_cluster_transformed))
                                    Reference_Line_Mask_Train.append(np.array(lane_valid, dtype=np.int16))
                                    Planned_Path_List_Train.append(ppt)
                                    Navi_Drive_List_Train.append(np.array([navi_drive], dtype=np.int16))
                                    Ego_State_List_Train.append(ego_state)
                                    Current_Context_List_Train.append(np.array([current_context], dtype=np.int16))
                                    Next_Context_List_Train.append(np.array([next_context], dtype=np.int16))
                                    Lookahead_Info_Cluster_List_Train.append(lookahead_info_cluster)

                                    ego_bias_sign = int(np.sign(-ego_pos_on_path[0]))
                                    target_bias_sign = int(np.sign(-ego_pos_on_path[0]))
                                    Lateral_Dist_To_Ego_Ref_Line_Train.append(np.array([np.abs(ego_pos_on_path[0])*ego_bias_sign],dtype=np.float16))
                                    Lateral_Dist_To_Target_Ref_Line_Train.append(np.array([np.abs(ego_pos_on_path[0])*target_bias_sign],dtype=np.float16))
                                    # bias符号(0-非负, 1->负)
                                    ego_bias_sign = 0 if ego_bias_sign>=0 else 1
                                    target_bias_sign = 0 if target_bias_sign>=0 else 1
                                    Ego_Bias_Pos_Neg_Info_Train.append(np.array([ego_bias_sign],dtype=np.int16))
                                    Target_Bias_Pos_Neg_Info_Train.append(np.array([target_bias_sign],dtype=np.int16))
                                    # bias大小比较(0--=, 1-->, 2--<)
                                    Abs_Bias_Compare_Info_Train.append(np.array([cmp(np.abs(ego_pos_on_path[0]), np.abs(ego_pos_on_path[0]))],dtype=np.int16))
                                    # 目标参考线的分布，比如00010
                                    Target_Reference_Line_Distrib_Train.append(np.array([0,0,1,0,0],dtype=np.int16))
                                    # ego, target参考线分布, 比如00110
                                    Ego_Target_Reference_Line_Distrib_Train.append(np.array([0,0,1,0,0],dtype=np.int16))
                                    # frenet s
                                    Frenet_S_On_Reference_Line_Train.append(s_list_on_planned_path)

                            else:
                                for has in heading_angle_settings:  # XXX: 3.自车朝向
                                    # 为了避免旋转操作后规划轨迹形态异常, 只取bias符号和heading符号一致的部分(此时规划轨迹仍然是合理的)
                                    if np.sign(bias)!=np.sign(has) and np.sign(bias)!=0 and np.sign(has)!=0:
                                        continue
                                    # 1.根据heading angle对lane_cluster进行旋转
                                    lane_cluster_transformed = []
                                    for j, lane in enumerate(lane_cluster_setting):
                                        if lane_valid_record[j]==1:
                                            lane_transformed = to_bev(lane, ego_pos, has)  # 输入参考线信息,放心使用,to_bev内部已经处理好了符号
                                        else:
                                            lane_transformed = lane
                                            
                                        # 只提取x>0部分(保留200个点)
                                        lane_transformed = lane_transformed[lane_transformed[:,1]>=0.0][0:200]

                                        # 一组5条经过平移旋转后的参考线输入(后续的normalization需要注意x,y是反的)
                                        lane_cluster_transformed.append(lane_transformed)

                                    # 2.根据heading angle对planned_path进行旋转
                                    planned_path_transformed = to_bev(planned_path, ego_pos, has)

                                    # XXX: 均匀采样出25个样本
                                    s_list_on_planned_path, lookahead_info_cluster = sample_required_num_items(25,
                                                                                                               i,
                                                                                                               ego_pos, 
                                                                                                               has, 
                                                                                                               v_ego, 
                                                                                                               total_s_list_on_planned_path, 
                                                                                                               total_lookahead_points)

                                    # 处理planned path, fix size
                                    ppt = planned_path_transformed[planned_path_transformed[:,1]>=0.0]
                                    if len(ppt)>=200:
                                        ppt = ppt[0:200]
                                    else:
                                        ppt = np.vstack((ppt, np.ones([200-len(ppt),2])*ppt[-1]))

                                    # context
                                    #current_context = context_types['lane_keep']        # 输入当前上下文
                                    #next_context = context_types['lane_keep']           # 输出下一时刻预测上下文
                                    # 尽可能增加current_context噪声使得对next context的预测鲁棒性更高
                                    for _, context in context_types.items():
                                        current_context = context
                                        next_context = context_types['lane_keep']           # 输出下一时刻预测上下文

                                        # 输入输出数据集
                                        Reference_Line_Cluster_List_Train.append(np.array(lane_cluster_transformed))
                                        Reference_Line_Mask_Train.append(np.array(lane_valid, dtype=np.int16))
                                        Planned_Path_List_Train.append(ppt)
                                        Navi_Drive_List_Train.append(np.array([navi_drive], dtype=np.int16))
                                        Ego_State_List_Train.append(ego_state)
                                        Current_Context_List_Train.append(np.array([current_context], dtype=np.int16))
                                        Next_Context_List_Train.append(np.array([next_context], dtype=np.int16))
                                        Lookahead_Info_Cluster_List_Train.append(lookahead_info_cluster)

                                        ego_bias_sign = int(np.sign(-bias))
                                        target_bias_sign = int(np.sign(-bias))
                                        Lateral_Dist_To_Ego_Ref_Line_Train.append(np.array([np.abs(bias)*ego_bias_sign],dtype=np.float16))
                                        Lateral_Dist_To_Target_Ref_Line_Train.append(np.array([np.abs(bias)*target_bias_sign],dtype=np.float16))
                                        # bias符号(0-非负, 1->负)
                                        ego_bias_sign = 0 if ego_bias_sign>=0 else 1
                                        target_bias_sign = 0 if target_bias_sign>=0 else 1
                                        Ego_Bias_Pos_Neg_Info_Train.append(np.array([ego_bias_sign],dtype=np.int16))
                                        Target_Bias_Pos_Neg_Info_Train.append(np.array([target_bias_sign],dtype=np.int16))
                                        # bias大小比较(0--=, 1-->, 2--<)
                                        Abs_Bias_Compare_Info_Train.append(np.array([cmp(np.abs(bias), np.abs(bias))],dtype=np.int16))
                                        # 目标参考线的分布，比如00010
                                        Target_Reference_Line_Distrib_Train.append(np.array([0,0,1,0,0],dtype=np.int16))
                                        # ego, target参考线分布, 比如00110
                                        Ego_Target_Reference_Line_Distrib_Train.append(np.array([0,0,1,0,0],dtype=np.int16))
                                        # frenet s
                                        Frenet_S_On_Reference_Line_Train.append(s_list_on_planned_path)


            ################################
            # 2.导航指令与道路有效性不冲突
            ################################
            else:
                if navi_drive==0: # 允许lk
                    for v_ego in v_ego_settings:    # XXX: 1.自车速度
                        # 自车的车头朝向可以在原车道内进行-theta_x~theta_x之间角度波动(仅用于参考planned path的坐标转换),左+右-
                        heading_angle_bound = heading_angle_bound_by_velocity(v_ego) # 自车速度约束可允许的最大车头朝向
                        heading_angle_step = (2.667/180)*np.pi
                        num_intervals = int(np.ceil(2*heading_angle_bound/heading_angle_step))
                        num_intervals = num_intervals+1 if num_intervals%2==0 else num_intervals
                        heading_angle_settings = np.linspace(-heading_angle_bound, heading_angle_bound, num_intervals).astype(np.float16)
                        if 0.0 not in list(heading_angle_settings):
                            heading_angle_settings = np.concatenate((heading_angle_settings, np.array([0.0], dtype=np.float16)), axis=0)
                        #print(v_ego, heading_angle_bound, num_intervals)

                        ego_state = np.array([v_ego, 0.0], dtype=np.float16)    # 输入自车信息(v, a)
                        # FIXME: 速度为零时等于零
                        s = v_ego*t_lk_max  # lk覆盖的纵向距离

                        for bias in lateral_bias_settings:  # XXX: 2.自车横向位置
                            # 在0号参考线坐标系下,即ref line 0所在的坐标系
                            ego_pos = np.array([bias, 0.0], dtype=np.float16) # -1.5~1.5m之间

                            # XXX: lk模式下planned path坐标已经是基于0号参考线坐标系
                            planned_path = generate_planned_path(s, bias, trend='lk')

                            # 一次性将lookahead_info_cluster和s_list_on_planned_path所涉及的所有项都准备好
                            total_s_list_on_planned_path, total_lookahead_points = withdraw_all_s_list_and_lookahead_points_needed(planned_path, 'lk')

                            for i in range(len(planned_path)):
                                # 这里需要根据planned path上不同位置计算对应的lookahead
                                if i>0:
                                    ego_pos_on_path = planned_path[i]
                                    vector = planned_path[i] - planned_path[i-1]
                                    heading_on_path = np.arctan(vector[0]/vector[1])

                                    # 1.根据heading angle对lane_cluster进行旋转
                                    lane_cluster_transformed = []
                                    for j, lane in enumerate(lane_cluster_setting):
                                        if lane_valid_record[j]==1:
                                            lane_transformed = to_bev(lane, ego_pos_on_path, heading_on_path)  # 输入参考线信息,放心使用,to_bev内部已经处理好了符号
                                        else:
                                            lane_transformed = lane

                                        # 只提取x>0部分(保留200个点)
                                        lane_transformed = lane_transformed[lane_transformed[:,1]>=0.0][0:200]

                                        # 一组5条经过平移旋转后的参考线输入(后续的normalization需要注意x,y是反的)
                                        lane_cluster_transformed.append(lane_transformed)

                                    # 2.根据heading angle对planned_path进行旋转
                                    planned_path_transformed = to_bev(planned_path, ego_pos_on_path, heading_on_path)

                                    # XXX: 均匀采样出25个样本
                                    s_list_on_planned_path, lookahead_info_cluster = sample_required_num_items(25,
                                                                                                               i,
                                                                                                               ego_pos_on_path, 
                                                                                                               heading_on_path, 
                                                                                                               v_ego, 
                                                                                                               total_s_list_on_planned_path, 
                                                                                                               total_lookahead_points)

                                    # 处理planned path, fix size
                                    ppt = planned_path_transformed[planned_path_transformed[:,1]>=0.0]
                                    if len(ppt)>=200:
                                        ppt = ppt[0:200]
                                    else:
                                        ppt = np.vstack((ppt, np.ones([200-len(ppt),2])*ppt[-1]))

                                    # context
                                    #current_context = context_types['lane_keep']        # 输入当前上下文
                                    #next_context = context_types['lane_keep']
                                    # 尽可能增加current_context噪声使得对next context的预测鲁棒性更高
                                    for _, context in context_types.items():
                                        current_context = context        # 输入当前上下文
                                        next_context = context_types['lane_keep']

                                        # 输入输出数据集
                                        Reference_Line_Cluster_List_Train.append(np.array(lane_cluster_transformed))
                                        Reference_Line_Mask_Train.append(np.array(lane_valid, dtype=np.int16))
                                        Planned_Path_List_Train.append(ppt)
                                        Navi_Drive_List_Train.append(np.array([navi_drive], dtype=np.int16))
                                        Ego_State_List_Train.append(ego_state)
                                        Current_Context_List_Train.append(np.array([current_context], dtype=np.int16))
                                        Next_Context_List_Train.append(np.array([next_context], dtype=np.int16))
                                        Lookahead_Info_Cluster_List_Train.append(lookahead_info_cluster)

                                        ego_bias_sign = int(np.sign(-ego_pos_on_path[0]))
                                        target_bias_sign = int(np.sign(-ego_pos_on_path[0]))
                                        Lateral_Dist_To_Ego_Ref_Line_Train.append(np.array([np.abs(ego_pos_on_path[0])*ego_bias_sign],dtype=np.float16))
                                        Lateral_Dist_To_Target_Ref_Line_Train.append(np.array([np.abs(ego_pos_on_path[0])*target_bias_sign],dtype=np.float16))
                                        # bias符号(0-非负, 1->负)
                                        ego_bias_sign = 0 if ego_bias_sign>=0 else 1
                                        target_bias_sign = 0 if target_bias_sign>=0 else 1
                                        Ego_Bias_Pos_Neg_Info_Train.append(np.array([ego_bias_sign],dtype=np.int16))
                                        Target_Bias_Pos_Neg_Info_Train.append(np.array([target_bias_sign],dtype=np.int16))
                                        # bias大小比较(0--=, 1-->, 2--<)
                                        Abs_Bias_Compare_Info_Train.append(np.array([cmp(np.abs(ego_pos_on_path[0]), np.abs(ego_pos_on_path[0]))],dtype=np.int16))
                                        # 目标参考线的分布，比如00010
                                        Target_Reference_Line_Distrib_Train.append(np.array([0,0,1,0,0],dtype=np.int16))
                                        # ego, target参考线分布, 比如00110
                                        Ego_Target_Reference_Line_Distrib_Train.append(np.array([0,0,1,0,0],dtype=np.int16))
                                        # frenet s
                                        Frenet_S_On_Reference_Line_Train.append(s_list_on_planned_path)

                                else:
                                    for has in heading_angle_settings:  # XXX: 3.自车朝向
                                        # 为了避免旋转操作后规划轨迹形态异常, 只取bias符号和heading符号一致的部分(此时规划轨迹仍然是合理的)
                                        if np.sign(bias)!=np.sign(has) and np.sign(bias)!=0 and np.sign(has)!=0:
                                            continue
                                        # 1.根据heading angle对lane_cluster进行旋转
                                        lane_cluster_transformed = []
                                        for j, lane in enumerate(lane_cluster_setting):
                                            if lane_valid_record[j]==1:
                                                lane_transformed = to_bev(lane, ego_pos, has)  # 输入参考线信息,放心使用,to_bev内部已经处理好了符号
                                            else:
                                                lane_transformed = lane
                                                
                                            # 只提取x>0部分(保留200个点)
                                            lane_transformed = lane_transformed[lane_transformed[:,1]>=0.0][0:200]

                                            # 一组5条经过平移旋转后的参考线输入(后续的normalization需要注意x,y是反的)
                                            lane_cluster_transformed.append(lane_transformed)

                                        # 2.根据heading angle对planned_path进行旋转
                                        planned_path_transformed = to_bev(planned_path, ego_pos, has)

                                        # XXX: 均匀采样出25个样本
                                        s_list_on_planned_path, lookahead_info_cluster = sample_required_num_items(25,
                                                                                                                   i,
                                                                                                                   ego_pos, 
                                                                                                                   has, 
                                                                                                                   v_ego, 
                                                                                                                   total_s_list_on_planned_path, 
                                                                                                                   total_lookahead_points)

                                        # 处理planned path, fix size
                                        ppt = planned_path_transformed[planned_path_transformed[:,1]>=0.0]
                                        if len(ppt)>=200:
                                            ppt = ppt[0:200]
                                        else:
                                            ppt = np.vstack((ppt, np.ones([200-len(ppt),2])*ppt[-1]))

                                        # context
                                        #current_context = context_types['lane_keep']        # 输入当前上下文
                                        #next_context = context_types['lane_keep']           # 输出下一时刻预测上下文
                                        # 尽可能增加current_context噪声使得对next context的预测鲁棒性更高
                                        for _, context in context_types.items():
                                            current_context = context        # 输入当前上下文
                                            next_context = context_types['lane_keep']           # 输出下一时刻预测上下文

                                            # 输入输出数据集
                                            Reference_Line_Cluster_List_Train.append(np.array(lane_cluster_transformed))
                                            Reference_Line_Mask_Train.append(np.array(lane_valid, dtype=np.int16))
                                            Planned_Path_List_Train.append(ppt)
                                            Navi_Drive_List_Train.append(np.array([navi_drive], dtype=np.int16))
                                            Ego_State_List_Train.append(ego_state)
                                            Current_Context_List_Train.append(np.array([current_context], dtype=np.int16))
                                            Next_Context_List_Train.append(np.array([next_context], dtype=np.int16))
                                            Lookahead_Info_Cluster_List_Train.append(lookahead_info_cluster)

                                            ego_bias_sign = int(np.sign(-bias))
                                            target_bias_sign = int(np.sign(-bias))
                                            Lateral_Dist_To_Ego_Ref_Line_Train.append(np.array([np.abs(bias)*ego_bias_sign],dtype=np.float16))
                                            Lateral_Dist_To_Target_Ref_Line_Train.append(np.array([np.abs(bias)*target_bias_sign],dtype=np.float16))
                                            # bias符号(0-非负, 1->负)
                                            ego_bias_sign = 0 if ego_bias_sign>=0 else 1
                                            target_bias_sign = 0 if target_bias_sign>=0 else 1
                                            Ego_Bias_Pos_Neg_Info_Train.append(np.array([ego_bias_sign],dtype=np.int16))
                                            Target_Bias_Pos_Neg_Info_Train.append(np.array([target_bias_sign],dtype=np.int16))
                                            # bias大小比较(0--=, 1-->, 2--<)
                                            Abs_Bias_Compare_Info_Train.append(np.array([cmp(np.abs(bias), np.abs(bias))],dtype=np.int16))
                                            # 目标参考线的分布，比如00010
                                            Target_Reference_Line_Distrib_Train.append(np.array([0,0,1,0,0],dtype=np.int16))
                                            # ego, target参考线分布, 比如00110
                                            Ego_Target_Reference_Line_Distrib_Train.append(np.array([0,0,1,0,0],dtype=np.int16))
                                            # frenet s
                                            Frenet_S_On_Reference_Line_Train.append(s_list_on_planned_path)


                elif navi_drive==1: # 允许llc
                    llc_lateral_bias_settings = lane_width - lateral_bias_settings_4llc

                    for v_ego in v_ego_settings:    # XXX: 1.自车速度
                        # 自车的车头朝向可以在原车道内进行-theta_x~theta_x之间角度波动(仅用于参考planned path的坐标转换),左+右-
                        heading_angle_bound = heading_angle_bound_by_velocity(v_ego) # 自车速度约束可允许的最大车头朝向
                        heading_angle_step = (2.667/180)*np.pi
                        num_intervals = int(np.ceil(2*heading_angle_bound/heading_angle_step))
                        num_intervals = num_intervals+1 if num_intervals%2==0 else num_intervals
                        heading_angle_settings = np.linspace(-heading_angle_bound, heading_angle_bound, num_intervals).astype(np.float16)
                        if 0.0 not in list(heading_angle_settings):
                            heading_angle_settings = np.concatenate((heading_angle_settings, np.array([0.0], dtype=np.float16)), axis=0)
                        #print(v_ego, heading_angle_bound, num_intervals)

                        ego_state = np.array([v_ego, 0.0], dtype=np.float16)    # 输入自车信息(v, a)
                        # FIXME: 速度为零时等于零
                        s = v_ego*t_lc_max  # lc覆盖的纵向距离

                        for k, bias in enumerate(llc_lateral_bias_settings):  # XXX: 2.自车横向位置
                            # 在0号参考线坐标系下,即ref line 0所在的坐标系
                            ego_pos = np.array([lateral_bias_settings_4llc[k], 0.0], dtype=np.float16) # -1.5~1.5m之间

                            # XXX: planned path基于0号参考线坐标系,往左换道的轨迹
                            planned_path = generate_planned_path(s, bias, trend='lc')
                            # 将planned_path(默认是以自车为坐标原点)转换到0号参考线坐标系，否则to_bev将出错
                            planned_path = transfer(planned_path, ego_pos)

                            # 一次性将lookahead_info_cluster和s_list_on_planned_path所涉及的所有项都准备好
                            total_s_list_on_planned_path, total_lookahead_points = withdraw_all_s_list_and_lookahead_points_needed(planned_path, 'llc')

                            # 第一帧current context默认是lk，最后一帧next context默认是lk，中间帧的current context和next context均是llc
                            # 第一帧heading默认是0.0，需要考虑heading_angle_settings制造变化，后续帧则直接从planned path上获取heading即可
                            for i in range(len(planned_path)):
                                # 这里需要根据planned path上不同位置计算对应的lookahead
                                if i>0:
                                    ego_pos_on_path = planned_path[i]
                                    vector = planned_path[i] - planned_path[i-1]
                                    heading_on_path = np.arctan(vector[0]/vector[1])

                                    # 1.根据heading angle对lane_cluster进行旋转
                                    lane_cluster_transformed = []
                                    for j, lane in enumerate(lane_cluster_setting):
                                        if lane_valid_record[j]==1:
                                            lane_transformed = to_bev(lane, ego_pos_on_path, heading_on_path)  # 输入参考线信息,放心使用,to_bev内部已经处理好了符号
                                        else:
                                            lane_transformed = lane

                                        # 只提取x>0部分(保留200个点)
                                        lane_transformed = lane_transformed[lane_transformed[:,1]>=0.0][0:200]

                                        # 一组5条经过平移旋转后的参考线输入(后续的normalization需要注意x,y是反的)
                                        lane_cluster_transformed.append(lane_transformed)

                                    # 2.根据heading angle对planned_path进行旋转
                                    planned_path_transformed = to_bev(planned_path, ego_pos_on_path, heading_on_path)

                                    # XXX: 均匀采样出25个样本
                                    s_list_on_planned_path, lookahead_info_cluster = sample_required_num_items(25,
                                                                                                               i,
                                                                                                               ego_pos_on_path, 
                                                                                                               heading_on_path, 
                                                                                                               v_ego, 
                                                                                                               total_s_list_on_planned_path, 
                                                                                                               total_lookahead_points)

                                    # 处理planned path, fix size
                                    ppt = planned_path_transformed[planned_path_transformed[:,1]>=0.0]
                                    if len(ppt)>=200:
                                        ppt = ppt[0:200]
                                    else:
                                        ppt = np.vstack((ppt, np.ones([200-len(ppt),2])*ppt[-1]))

                                    ## XXX: 只要navi drive不是lk, 则一直保持lc状态
                                    # context
                                    #current_context = context_types['left_lane_change']        # 输入当前上下文
                                    # 输出下一时刻预测上下文
                                    #if i!=len(planned_path)-1:
                                    #    next_context = context_types['left_lane_change']
                                    #else:
                                    #    next_context = context_types['lane_keep']


                                    # 时刻计算当前位置到ego和target参考线的横向距离
                                    abs_d_ego = np.abs(planned_path[i][0])
                                    abs_d_target = np.abs(lane_width-planned_path[i][0])

                                    # 尽可能增加current_context噪声使得对next context的预测鲁棒性更高
                                    for _, context in context_types.items():
                                        current_context = context
                                        next_context = context_types['left_lane_change']

                                        # 换道过程中，若abs_d_target低于lc_to_lk_threshold(暂定1.5米)意味着bev视野内ego dominated(0号)参考线要得到更新
                                        if abs_d_target>lc_to_lk_threshold:
                                            updated_lane_valid = lane_valid # 无需更新ego dominated参考线
                                            updated_lane_cluster_transformed = lane_cluster_transformed # 无需更新lane_cluster_transformed
                                        else:
                                            updated_lane_valid = copy.deepcopy(lane_valid)
                                            updated_lane_cluster_transformed = copy.deepcopy(lane_cluster_transformed)
                                            # 向左换道，丢弃(最先看不到)最右侧的lane
                                            updated_lane_valid.pop(-1)
                                            updated_lane_cluster_transformed.pop(-1)
                                            # 判断最左侧是否是有效车道
                                            if updated_lane_valid[0]==0:
                                                updated_lane_valid = [0] + updated_lane_valid # 最左侧添加一条空车道是最合理的假设
                                                updated_lane_cluster_transformed = [invalid_ref_line[0:200]] + updated_lane_cluster_transformed # 最左侧添加一条空车道
                                            else:
                                                # 最右侧既可以添加一条有效车道, 也可以添加一条空车道
                                                # FIXME: 受制于电脑内存限制, 采取抽签方式决定是添加有效车道, 还是空车道
                                                if np.random.rand()>=0.5:
                                                    updated_lane_valid = [0] + updated_lane_valid # 抽签决定最右侧添加一条空车道
                                                    updated_lane_cluster_transformed = [invalid_ref_line[0:200]] + updated_lane_cluster_transformed
                                                else:
                                                    updated_lane_valid = [1] + updated_lane_valid # 抽签决定最右侧添加一条有效车道
                                                    extra_lane = transfer(updated_lane_cluster_transformed[0], np.array([lane_width, 0.0], dtype=np.float16))
                                                    updated_lane_cluster_transformed = [extra_lane] + updated_lane_cluster_transformed

                                        # 输入输出数据集
                                        Reference_Line_Cluster_List_Train.append(np.array(updated_lane_cluster_transformed))
                                        Reference_Line_Mask_Train.append(np.array(updated_lane_valid, dtype=np.int16))
                                        Planned_Path_List_Train.append(ppt)
                                        Navi_Drive_List_Train.append(np.array([navi_drive], dtype=np.int16))
                                        Ego_State_List_Train.append(ego_state)
                                        Current_Context_List_Train.append(np.array([current_context], dtype=np.int16))
                                        Next_Context_List_Train.append(np.array([next_context], dtype=np.int16))
                                        Lookahead_Info_Cluster_List_Train.append(lookahead_info_cluster)

                                        ego_bias_sign = int(np.sign(-planned_path[i][0]))
                                        target_bias_sign = int(np.sign(lane_width-planned_path[i][0]))
                                        Lateral_Dist_To_Ego_Ref_Line_Train.append(np.array([abs_d_ego*ego_bias_sign],dtype=np.float16))
                                        Lateral_Dist_To_Target_Ref_Line_Train.append(np.array([abs_d_target*target_bias_sign],dtype=np.float16))
                                        # bias符号(0-非负, 1->负)
                                        ego_bias_sign = 0 if ego_bias_sign>=0 else 1
                                        target_bias_sign = 0 if target_bias_sign>=0 else 1
                                        Ego_Bias_Pos_Neg_Info_Train.append(np.array([ego_bias_sign],dtype=np.int16))
                                        Target_Bias_Pos_Neg_Info_Train.append(np.array([target_bias_sign],dtype=np.int16))
                                        # bias大小比较(0--=, 1-->, 2--<)
                                        Abs_Bias_Compare_Info_Train.append(np.array([cmp(abs_d_ego, abs_d_target)],dtype=np.int16))
                                        ###############################################
                                        # ego, target参考线在参考线cluster中的分布情况
                                        ###############################################
                                        if abs_d_target>lc_to_lk_threshold: # 换道过程中，若abs_d_target低于1.5米意味着bev视野内ego dominated(0号)参考线要得到更新
                                            trl_distr = np.array([0,1,0,0,0],dtype=np.int16)
                                            etrl_distr = np.array([0,1,1,0,0],dtype=np.int16)
                                        else:
                                            trl_distr = np.array([0,0,1,0,0],dtype=np.int16)
                                            etrl_distr = np.array([0,0,1,1,0],dtype=np.int16)
                                        # 目标参考线的分布，比如00010
                                        Target_Reference_Line_Distrib_Train.append(trl_distr)
                                        # ego, target参考线分布, 比如00110
                                        Ego_Target_Reference_Line_Distrib_Train.append(etrl_distr)
                                        # frenet s
                                        Frenet_S_On_Reference_Line_Train.append(s_list_on_planned_path)

                                else:
                                    for has in heading_angle_settings:  # XXX: 3.自车朝向
                                        # 为了避免旋转操作后规划轨迹形态异常, 只取heading合理的部分(此时规划轨迹仍然是合理的)
                                        if np.sign(has)==1:
                                            continue
                                        # 1.根据heading angle对lane_cluster进行旋转
                                        lane_cluster_transformed = []
                                        for j, lane in enumerate(lane_cluster_setting):
                                            if lane_valid_record[j]==1:
                                                lane_transformed = to_bev(lane, ego_pos, has)  # 输入参考线信息,放心使用,to_bev内部已经处理好了符号
                                            else:
                                                lane_transformed = lane
                                                
                                            # 只提取x>0部分(保留200个点)
                                            lane_transformed = lane_transformed[lane_transformed[:,1]>=0.0][0:200]

                                            # 一组5条经过平移旋转后的参考线输入(后续的normalization需要注意x,y是反的)
                                            lane_cluster_transformed.append(lane_transformed)

                                        # 2.根据heading angle对planned_path进行旋转
                                        planned_path_transformed = to_bev(planned_path, ego_pos, has)

                                        # XXX: 均匀采样出25个样本
                                        s_list_on_planned_path, lookahead_info_cluster = sample_required_num_items(25,
                                                                                                                   i,
                                                                                                                   ego_pos, 
                                                                                                                   has, 
                                                                                                                   v_ego, 
                                                                                                                   total_s_list_on_planned_path, 
                                                                                                                   total_lookahead_points)

                                        # 处理planned path, fix size
                                        ppt = planned_path_transformed[planned_path_transformed[:,1]>=0.0]
                                        if len(ppt)>=200:
                                            ppt = ppt[0:200]
                                        else:
                                            ppt = np.vstack((ppt, np.ones([200-len(ppt),2])*ppt[-1]))

                                        # context
                                        #current_context = context_types['lane_keep']        # 输入当前上下文
                                        #next_context = context_types['left_lane_change']           # 输出下一时刻预测上下文

                                        # 尽可能增加current_context噪声使得对next context的预测鲁棒性更高
                                        for _, context in context_types.items():
                                            current_context = context        # 输入当前上下文
                                            next_context = context_types['left_lane_change']           # 输出下一时刻预测上下文

                                            # 输入输出数据集
                                            Reference_Line_Cluster_List_Train.append(np.array(lane_cluster_transformed))
                                            Reference_Line_Mask_Train.append(np.array(lane_valid, dtype=np.int16))
                                            Planned_Path_List_Train.append(ppt)
                                            Navi_Drive_List_Train.append(np.array([navi_drive], dtype=np.int16))
                                            Ego_State_List_Train.append(ego_state)
                                            Current_Context_List_Train.append(np.array([current_context], dtype=np.int16))
                                            Next_Context_List_Train.append(np.array([next_context], dtype=np.int16))
                                            Lookahead_Info_Cluster_List_Train.append(lookahead_info_cluster)

                                            ego_bias_sign = int(np.sign(-planned_path[i][0]))
                                            target_bias_sign = int(np.sign(lane_width-planned_path[i][0]))
                                            Lateral_Dist_To_Ego_Ref_Line_Train.append(np.array([np.abs(planned_path[i][0])*ego_bias_sign],dtype=np.float16))
                                            Lateral_Dist_To_Target_Ref_Line_Train.append(np.array([np.abs(lane_width-planned_path[i][0])*target_bias_sign],dtype=np.float16))
                                            # bias符号(0-非负, 1->负)
                                            ego_bias_sign = 0 if ego_bias_sign>=0 else 1
                                            target_bias_sign = 0 if target_bias_sign>=0 else 1
                                            Ego_Bias_Pos_Neg_Info_Train.append(np.array([ego_bias_sign],dtype=np.int16))
                                            Target_Bias_Pos_Neg_Info_Train.append(np.array([target_bias_sign],dtype=np.int16))
                                            # bias大小比较(0--=, 1-->, 2--<)
                                            Abs_Bias_Compare_Info_Train.append(np.array([cmp(np.abs(planned_path[i][0]), np.abs(lane_width-planned_path[i][0]))],dtype=np.int16))
                                            # 目标参考线的分布，比如00010
                                            Target_Reference_Line_Distrib_Train.append(np.array([0,1,0,0,0],dtype=np.int16))
                                            # ego, target参考线分布, 比如00110
                                            Ego_Target_Reference_Line_Distrib_Train.append(np.array([0,1,1,0,0],dtype=np.int16))
                                            # frenet s
                                            Frenet_S_On_Reference_Line_Train.append(s_list_on_planned_path)
                    
                else: # 允许rlc
                    rlc_lateral_bias_settings = -(lateral_bias_settings_4rlc + lane_width)

                    for v_ego in v_ego_settings:    # XXX: 1.自车速度
                        # 自车的车头朝向可以在原车道内进行-theta_x~theta_x之间角度波动(仅用于参考planned path的坐标转换),左+右-
                        heading_angle_bound = heading_angle_bound_by_velocity(v_ego) # 自车速度约束可允许的最大车头朝向
                        heading_angle_step = (2.667/180)*np.pi
                        num_intervals = int(np.ceil(2*heading_angle_bound/heading_angle_step))
                        num_intervals = num_intervals+1 if num_intervals%2==0 else num_intervals
                        heading_angle_settings = np.linspace(-heading_angle_bound, heading_angle_bound, num_intervals).astype(np.float16)
                        if 0.0 not in list(heading_angle_settings):
                            heading_angle_settings = np.concatenate((heading_angle_settings, np.array([0.0], dtype=np.float16)), axis=0)
                        #print(v_ego, heading_angle_bound, num_intervals)

                        ego_state = np.array([v_ego, 0.0], dtype=np.float16)    # 输入自车信息(v, a)
                        # FIXME: 速度为零时等于零
                        s = v_ego*t_lc_max  # lc覆盖的纵向距离

                        for k, bias in enumerate(rlc_lateral_bias_settings):  # XXX: 2.自车横向位置
                            # 在0号参考线坐标系下,即ref line 0所在的坐标系
                            ego_pos = np.array([lateral_bias_settings_4rlc[k], 0.0], dtype=np.float16) # -1.5~1.5m之间

                            # XXX: planned path基于0号参考线坐标系,往左换道的轨迹
                            planned_path = generate_planned_path(s, bias, trend='lc')
                            # 将planned_path(默认是以自车为坐标原点)转换到0号参考线坐标系，否则to_bev将出错
                            planned_path = transfer(planned_path, ego_pos)

                            # 一次性将lookahead_info_cluster和s_list_on_planned_path所涉及的所有项都准备好
                            total_s_list_on_planned_path, total_lookahead_points = withdraw_all_s_list_and_lookahead_points_needed(planned_path, 'rlc')

                            # 第一帧current context默认是lk，最后一帧next context默认是lk，中间帧的current context和next context均是llc
                            # 第一帧heading默认是0.0，需要考虑heading_angle_settings制造变化，后续帧则直接从planned path上获取heading即可
                            for i in range(len(planned_path)):
                                # 这里需要根据planned path上不同位置计算对应的lookahead
                                if i>0:
                                    ego_pos_on_path = planned_path[i]
                                    vector = planned_path[i] - planned_path[i-1]
                                    heading_on_path = np.arctan(vector[0]/vector[1])

                                    # 1.根据heading angle对lane_cluster进行旋转
                                    lane_cluster_transformed = []
                                    for j, lane in enumerate(lane_cluster_setting):
                                        if lane_valid_record[j]==1:
                                            lane_transformed = to_bev(lane, ego_pos_on_path, heading_on_path)  # 输入参考线信息,放心使用,to_bev内部已经处理好了符号
                                        else:
                                            lane_transformed = lane

                                        # 只提取x>0部分(保留200个点)
                                        lane_transformed = lane_transformed[lane_transformed[:,1]>=0.0][0:200]

                                        # 一组5条经过平移旋转后的参考线输入(后续的normalization需要注意x,y是反的)
                                        lane_cluster_transformed.append(lane_transformed)

                                    # 2.根据heading angle对planned_path进行旋转
                                    planned_path_transformed = to_bev(planned_path, ego_pos_on_path, heading_on_path)

                                    # XXX: 均匀采样出25个样本
                                    s_list_on_planned_path, lookahead_info_cluster = sample_required_num_items(25,
                                                                                                               i,
                                                                                                               ego_pos_on_path, 
                                                                                                               heading_on_path, 
                                                                                                               v_ego, 
                                                                                                               total_s_list_on_planned_path, 
                                                                                                               total_lookahead_points)

                                    # 处理planned path, fix size
                                    ppt = planned_path_transformed[planned_path_transformed[:,1]>=0.0]
                                    if len(ppt)>=200:
                                        ppt = ppt[0:200]
                                    else:
                                        ppt = np.vstack((ppt, np.ones([200-len(ppt),2])*ppt[-1]))

                                    # context
                                    #current_context = context_types['right_lane_change']        # 输入当前上下文
                                    # 输出下一时刻预测上下文
                                    #if i!=len(planned_path)-1:
                                    #    next_context = context_types['right_lane_change']
                                    #else:
                                    #    next_context = context_types['lane_keep']


                                    # 时刻计算当前位置到ego和target参考线的横向距离
                                    abs_d_ego = np.abs(planned_path[i][0])
                                    abs_d_target = np.abs(lane_width+planned_path[i][0])

                                    # 尽可能增加current_context噪声使得对next context的预测鲁棒性更高
                                    for _, context in context_types.items():
                                        current_context = context
                                        next_context = context_types['right_lane_change']

                                        # 换道过程中，若abs_d_target低于lc_to_lk_threshold(暂定1.5米)意味着bev视野内ego dominated(0号)参考线要得到更新
                                        if abs_d_target>lc_to_lk_threshold:
                                            updated_lane_valid = lane_valid # 无需更新ego dominated参考线
                                            updated_lane_cluster_transformed = lane_cluster_transformed # 无需更新lane_cluster_transformed
                                        else:
                                            updated_lane_valid = copy.deepcopy(lane_valid)
                                            updated_lane_cluster_transformed = copy.deepcopy(lane_cluster_transformed)
                                            # 向右换道，丢弃(最先看不到)最左侧的lane
                                            updated_lane_valid.pop(0)
                                            updated_lane_cluster_transformed.pop(0)
                                            # 判断最右侧是是否是有效车道
                                            if updated_lane_valid[-1]==0:
                                                updated_lane_valid.append(0) # 最右侧添加一条空车道是最合理的假设
                                                updated_lane_cluster_transformed.append(invalid_ref_line[0:200])
                                            else:
                                                # 最右侧既可以添加一条有效车道, 也可以添加一条空车道
                                                # FIXME: 受制于电脑内存限制, 采取抽签方式决定是使用有效车道, 还是空车道
                                                if np.random.rand()>=0.5:
                                                    updated_lane_valid.append(0) # 抽签决定最右侧添加一条空车道
                                                    updated_lane_cluster_transformed.append(invalid_ref_line[0:200])
                                                else:
                                                    updated_lane_valid.append(1) # 抽签决定最右侧添加一条有效车道
                                                    extra_lane = transfer(updated_lane_cluster_transformed[-1], np.array([-lane_width, 0.0], dtype=np.float16))
                                                    updated_lane_cluster_transformed.append(extra_lane)

                                        # 输入输出数据集
                                        Reference_Line_Cluster_List_Train.append(np.array(updated_lane_cluster_transformed))
                                        Reference_Line_Mask_Train.append(np.array(updated_lane_valid, dtype=np.int16))
                                        Planned_Path_List_Train.append(ppt)
                                        Navi_Drive_List_Train.append(np.array([navi_drive], dtype=np.int16))
                                        Ego_State_List_Train.append(ego_state)
                                        Current_Context_List_Train.append(np.array([current_context], dtype=np.int16))
                                        Next_Context_List_Train.append(np.array([next_context], dtype=np.int16))
                                        Lookahead_Info_Cluster_List_Train.append(lookahead_info_cluster)

                                        ego_bias_sign = int(np.sign(-planned_path[i][0]))
                                        target_bias_sign = int(np.sign(-(lane_width+planned_path[i][0])))
                                        Lateral_Dist_To_Ego_Ref_Line_Train.append(np.array([abs_d_ego*ego_bias_sign],dtype=np.float16))
                                        Lateral_Dist_To_Target_Ref_Line_Train.append(np.array([abs_d_target*target_bias_sign],dtype=np.float16))
                                        # bias符号(0-非负, 1->负)
                                        ego_bias_sign = 0 if ego_bias_sign>=0 else 1
                                        target_bias_sign = 0 if target_bias_sign>=0 else 1
                                        Ego_Bias_Pos_Neg_Info_Train.append(np.array([ego_bias_sign],dtype=np.int16))
                                        Target_Bias_Pos_Neg_Info_Train.append(np.array([target_bias_sign],dtype=np.int16))
                                        # bias大小比较(0--=, 1-->, 2--<)
                                        Abs_Bias_Compare_Info_Train.append(np.array([cmp(abs_d_ego, abs_d_target)],dtype=np.int16))
                                        ###############################################
                                        # ego, target参考线在参考线cluster中的分布情况
                                        ###############################################
                                        if abs_d_target>lc_to_lk_threshold: # 换道过程中，若abs_d_target低于1.5米意味着bev视野内ego dominated(0号)参考线要得到更新
                                            trl_distr = np.array([0,0,0,1,0],dtype=np.int16)
                                            etrl_distr = np.array([0,0,1,1,0],dtype=np.int16)
                                        else:
                                            trl_distr = np.array([0,0,1,0,0],dtype=np.int16)
                                            etrl_distr = np.array([0,1,1,0,0],dtype=np.int16)
                                        # 目标参考线的分布，比如00010
                                        Target_Reference_Line_Distrib_Train.append(trl_distr)
                                        # ego, target参考线分布, 比如00110
                                        Ego_Target_Reference_Line_Distrib_Train.append(etrl_distr)
                                        # frenet s
                                        Frenet_S_On_Reference_Line_Train.append(s_list_on_planned_path)

                                else:
                                    for has in heading_angle_settings:  # XXX: 3.自车朝向
                                        # 为了避免旋转操作后规划轨迹形态异常, 只取heading合理的部分(此时规划轨迹仍然是合理的)
                                        if np.sign(has)==-1:
                                            continue
                                        # 1.根据heading angle对lane_cluster进行旋转
                                        lane_cluster_transformed = []
                                        for j, lane in enumerate(lane_cluster_setting):
                                            if lane_valid_record[j]==1:
                                                lane_transformed = to_bev(lane, ego_pos, has)  # 输入参考线信息,放心使用,to_bev内部已经处理好了符号
                                            else:
                                                lane_transformed = lane
                                                
                                            # 只提取x>0部分(保留200个点)
                                            lane_transformed = lane_transformed[lane_transformed[:,1]>=0.0][0:200]

                                            # 一组5条经过平移旋转后的参考线输入(后续的normalization需要注意x,y是反的)
                                            lane_cluster_transformed.append(lane_transformed)

                                        # 2.根据heading angle对planned_path进行旋转
                                        planned_path_transformed = to_bev(planned_path, ego_pos, has)

                                        # XXX: 均匀采样出25个样本
                                        s_list_on_planned_path, lookahead_info_cluster = sample_required_num_items(25,
                                                                                                                   i,
                                                                                                                   ego_pos, 
                                                                                                                   has, 
                                                                                                                   v_ego, 
                                                                                                                   total_s_list_on_planned_path, 
                                                                                                                   total_lookahead_points)

                                        # 处理planned path, fix size
                                        ppt = planned_path_transformed[planned_path_transformed[:,1]>=0.0]
                                        if len(ppt)>=200:
                                            ppt = ppt[0:200]
                                        else:
                                            ppt = np.vstack((ppt, np.ones([200-len(ppt),2])*ppt[-1]))

                                        # context
                                        #current_context = context_types['lane_keep']        # 输入当前上下文
                                        #next_context = context_types['right_lane_change']           # 输出下一时刻预测上下文

                                        # 尽可能增加current_context噪声使得对next context的预测鲁棒性更高
                                        for _, context in context_types.items():
                                            current_context = context
                                            next_context = context_types['right_lane_change']
                                        
                                            # 输入输出数据集
                                            Reference_Line_Cluster_List_Train.append(np.array(lane_cluster_transformed))
                                            Reference_Line_Mask_Train.append(np.array(lane_valid, dtype=np.int16))
                                            Planned_Path_List_Train.append(ppt)
                                            Navi_Drive_List_Train.append(np.array([navi_drive], dtype=np.int16))
                                            Ego_State_List_Train.append(ego_state)
                                            Current_Context_List_Train.append(np.array([current_context], dtype=np.int16))
                                            Next_Context_List_Train.append(np.array([next_context], dtype=np.int16))
                                            Lookahead_Info_Cluster_List_Train.append(lookahead_info_cluster)

                                            ego_bias_sign = int(np.sign(-planned_path[i][0]))
                                            target_bias_sign = int(np.sign(-(lane_width+planned_path[i][0])))
                                            Lateral_Dist_To_Ego_Ref_Line_Train.append(np.array([np.abs(planned_path[i][0])*ego_bias_sign],dtype=np.float16))
                                            Lateral_Dist_To_Target_Ref_Line_Train.append(np.array([np.abs(lane_width+planned_path[i][0])*target_bias_sign],dtype=np.float16))
                                            # bias符号(0-非负, 1->负)
                                            ego_bias_sign = 0 if ego_bias_sign>=0 else 1
                                            target_bias_sign = 0 if target_bias_sign>=0 else 1
                                            Ego_Bias_Pos_Neg_Info_Train.append(np.array([ego_bias_sign],dtype=np.int16))
                                            Target_Bias_Pos_Neg_Info_Train.append(np.array([target_bias_sign],dtype=np.int16))
                                            # bias大小比较(0--=, 1-->, 2--<)
                                            Abs_Bias_Compare_Info_Train.append(np.array([cmp(np.abs(planned_path[i][0]), np.abs(lane_width+planned_path[i][0]))],dtype=np.int16))
                                            # 目标参考线的分布，比如00010
                                            Target_Reference_Line_Distrib_Train.append(np.array([0,0,0,1,0],dtype=np.int16))
                                            # ego, target参考线分布, 比如00110
                                            Ego_Target_Reference_Line_Distrib_Train.append(np.array([0,0,1,1,0],dtype=np.int16))
                                            # frenet s
                                            Frenet_S_On_Reference_Line_Train.append(s_list_on_planned_path)


    assert len(Reference_Line_Cluster_List_Train)==len(Reference_Line_Mask_Train) and \
           len(Reference_Line_Mask_Train)==len(Planned_Path_List_Train) and \
           len(Planned_Path_List_Train)==len(Navi_Drive_List_Train) and \
           len(Navi_Drive_List_Train)==len(Ego_State_List_Train) and \
           len(Ego_State_List_Train)==len(Current_Context_List_Train) and \
           len(Current_Context_List_Train)==len(Next_Context_List_Train) and \
           len(Next_Context_List_Train)==len(Lookahead_Info_Cluster_List_Train) and \
           len(Lookahead_Info_Cluster_List_Train)==len(Lateral_Dist_To_Ego_Ref_Line_Train) and \
           len(Lateral_Dist_To_Ego_Ref_Line_Train)==len(Lateral_Dist_To_Target_Ref_Line_Train) and \
           len(Lateral_Dist_To_Target_Ref_Line_Train)==len(Ego_Bias_Pos_Neg_Info_Train) and \
           len(Ego_Bias_Pos_Neg_Info_Train)==len(Target_Bias_Pos_Neg_Info_Train) and \
           len(Target_Bias_Pos_Neg_Info_Train)==len(Abs_Bias_Compare_Info_Train) and \
           len(Abs_Bias_Compare_Info_Train)==len(Target_Reference_Line_Distrib_Train) and \
           len(Target_Reference_Line_Distrib_Train)==len(Ego_Target_Reference_Line_Distrib_Train) and \
           len(Ego_Target_Reference_Line_Distrib_Train)==len(Frenet_S_On_Reference_Line_Train), "输入输出数据的总数量出现不一致错误!"

    # 随机采样一些样本用于测试, 保留的index
    total_num = len(Reference_Line_Cluster_List_Train)
    test_indexes = random.choices(range(total_num), k=int(total_num*test_split))
    test_indexes = set(test_indexes) # 去重
    test_indexes = list(test_indexes)

    for idx in tqdm(test_indexes):
        Reference_Line_Cluster_List_Test.append(Reference_Line_Cluster_List_Train[idx])
        Reference_Line_Mask_Test.append(Reference_Line_Mask_Train[idx])
        Planned_Path_List_Test.append(Planned_Path_List_Train[idx])
        Navi_Drive_List_Test.append(Navi_Drive_List_Train[idx])
        Ego_State_List_Test.append(Ego_State_List_Train[idx])
        Current_Context_List_Test.append(Current_Context_List_Train[idx])
        Next_Context_List_Test.append(Next_Context_List_Train[idx])
        Lookahead_Info_Cluster_List_Test.append(Lookahead_Info_Cluster_List_Train[idx])
        Lateral_Dist_To_Ego_Ref_Line_Test.append(Lateral_Dist_To_Ego_Ref_Line_Train[idx])
        Lateral_Dist_To_Target_Ref_Line_Test.append(Lateral_Dist_To_Target_Ref_Line_Train[idx])

        Ego_Bias_Pos_Neg_Info_Test.append(Ego_Bias_Pos_Neg_Info_Train[idx])
        Target_Bias_Pos_Neg_Info_Test.append(Target_Bias_Pos_Neg_Info_Train[idx])
        Abs_Bias_Compare_Info_Test.append(Abs_Bias_Compare_Info_Train[idx])
        Target_Reference_Line_Distrib_Test.append(Target_Reference_Line_Distrib_Train[idx])
        Ego_Target_Reference_Line_Distrib_Test.append(Ego_Target_Reference_Line_Distrib_Train[idx])
        Frenet_S_On_Reference_Line_Test.append(Frenet_S_On_Reference_Line_Train[idx])

    # remove those for test
    test_indexes.sort()
    rm_times = 0
    for idx in tqdm(test_indexes):
        Reference_Line_Cluster_List_Train.pop(idx-rm_times)
        Reference_Line_Mask_Train.pop(idx-rm_times)
        Planned_Path_List_Train.pop(idx-rm_times)
        Navi_Drive_List_Train.pop(idx-rm_times)
        Ego_State_List_Train.pop(idx-rm_times)
        Current_Context_List_Train.pop(idx-rm_times)
        Next_Context_List_Train.pop(idx-rm_times)
        Lookahead_Info_Cluster_List_Train.pop(idx-rm_times)
        Lateral_Dist_To_Ego_Ref_Line_Train.pop(idx-rm_times)
        Lateral_Dist_To_Target_Ref_Line_Train.pop(idx-rm_times)

        Ego_Bias_Pos_Neg_Info_Train.pop(idx-rm_times)
        Target_Bias_Pos_Neg_Info_Train.pop(idx-rm_times)
        Abs_Bias_Compare_Info_Train.pop(idx-rm_times)
        Target_Reference_Line_Distrib_Train.pop(idx-rm_times)
        Ego_Target_Reference_Line_Distrib_Train.pop(idx-rm_times)
        Frenet_S_On_Reference_Line_Train.pop(idx-rm_times)

        rm_times += 1



if __name__ == '__main__':
    data_generator(Reference_Line_Cluster_List_Train,
                   Reference_Line_Mask_Train,
                   Planned_Path_List_Train,
                   Navi_Drive_List_Train,
                   Ego_State_List_Train,
                   Current_Context_List_Train,
                   Next_Context_List_Train,
                   Lookahead_Info_Cluster_List_Train,
                   Lateral_Dist_To_Ego_Ref_Line_Train,
                   Lateral_Dist_To_Target_Ref_Line_Train,
                   Ego_Bias_Pos_Neg_Info_Train,
                   Target_Bias_Pos_Neg_Info_Train,
                   Abs_Bias_Compare_Info_Train,
                   Target_Reference_Line_Distrib_Train,
                   Ego_Target_Reference_Line_Distrib_Train,
                   Frenet_S_On_Reference_Line_Train,
                   Reference_Line_Cluster_List_Test,
                   Reference_Line_Mask_Test,
                   Planned_Path_List_Test,
                   Navi_Drive_List_Test,
                   Ego_State_List_Test,
                   Current_Context_List_Test,
                   Next_Context_List_Test,
                   Lookahead_Info_Cluster_List_Test,
                   Lateral_Dist_To_Ego_Ref_Line_Test,
                   Lateral_Dist_To_Target_Ref_Line_Test,
                   Ego_Bias_Pos_Neg_Info_Test,
                   Target_Bias_Pos_Neg_Info_Test,
                   Abs_Bias_Compare_Info_Test,
                   Target_Reference_Line_Distrib_Test,
                   Ego_Target_Reference_Line_Distrib_Test,
                   Frenet_S_On_Reference_Line_Test
                   )

    # shuffle data
    train_data_indexes = list(range(len(Reference_Line_Cluster_List_Train)))
    test_data_indexes = list(range(len(Reference_Line_Cluster_List_Test)))

    # completely shuffled
    for i in range(100):
        random.shuffle(train_data_indexes)
        random.shuffle(test_data_indexes)
    print("total train: {}, total test: {}".format(len(train_data_indexes), len(test_data_indexes)))

    ##
    num_blk = 10 # 分成10份写文件
    avg_num_per_block = int(len(Reference_Line_Cluster_List_Train)/num_blk)

    #############
    # save train
    #############
    Reference_Line_Cluster_List_Train = np.array(Reference_Line_Cluster_List_Train, dtype=np.float16)
    Reference_Line_Mask_Train = np.array(Reference_Line_Mask_Train, dtype=np.int16)
    Planned_Path_List_Train = np.array(Planned_Path_List_Train, dtype=np.float16)
    Navi_Drive_List_Train = np.array(Navi_Drive_List_Train, dtype=np.int16)
    Ego_State_List_Train = np.array(Ego_State_List_Train, dtype=np.int16)
    Current_Context_List_Train = np.array(Current_Context_List_Train, dtype=np.int16)
    Next_Context_List_Train = np.array(Next_Context_List_Train, dtype=np.int16)
    Lookahead_Info_Cluster_List_Train = np.array(Lookahead_Info_Cluster_List_Train, dtype=np.float16)
    Lateral_Dist_To_Ego_Ref_Line_Train = np.array(Lateral_Dist_To_Ego_Ref_Line_Train, dtype=np.float16)
    Lateral_Dist_To_Target_Ref_Line_Train = np.array(Lateral_Dist_To_Target_Ref_Line_Train, dtype=np.float16)
    #
    Ego_Bias_Pos_Neg_Info_Train = np.array(Ego_Bias_Pos_Neg_Info_Train, dtype=np.int16)
    Target_Bias_Pos_Neg_Info_Train = np.array(Target_Bias_Pos_Neg_Info_Train, dtype=np.int16)
    Abs_Bias_Compare_Info_Train = np.array(Abs_Bias_Compare_Info_Train, dtype=np.int16)
    Target_Reference_Line_Distrib_Train = np.array(Target_Reference_Line_Distrib_Train, dtype=np.int16)
    Ego_Target_Reference_Line_Distrib_Train = np.array(Ego_Target_Reference_Line_Distrib_Train, dtype=np.int16)
    Frenet_S_On_Reference_Line_Train = np.array(Frenet_S_On_Reference_Line_Train, dtype=np.float16)

    print("Start to write npy files, please wait...")
    for i in tqdm(range(num_blk)):
        reference_line_train_filename = 'reference_line_{}.npy'.format(i)
        reference_line_train_file = os.path.join('./converted_dataset/diverse_train', reference_line_train_filename)

        reference_line_mask_train_filename = 'reference_line_mask_{}.npy'.format(i)
        reference_line_mask_train_file = os.path.join('./converted_dataset/diverse_train', reference_line_mask_train_filename)

        planned_path_train_filename = 'planned_path_{}.npy'.format(i)
        planned_path_train_file = os.path.join('./converted_dataset/diverse_train', planned_path_train_filename)

        navi_drive_train_filename = 'navi_drive_{}.npy'.format(i)
        navi_drive_train_file = os.path.join('./converted_dataset/diverse_train', navi_drive_train_filename)

        ego_state_train_filename = 'ego_state_{}.npy'.format(i)
        ego_state_train_file = os.path.join('./converted_dataset/diverse_train', ego_state_train_filename)

        current_context_train_filename = 'current_context_{}.npy'.format(i)
        current_context_train_file = os.path.join('./converted_dataset/diverse_train', current_context_train_filename)

        next_context_train_filename = 'next_context_{}.npy'.format(i)
        next_context_train_file = os.path.join('./converted_dataset/diverse_train', next_context_train_filename)

        lookahead_info_train_filename = 'lookahead_info_{}.npy'.format(i)
        lookahead_info_train_file = os.path.join('./converted_dataset/diverse_train', lookahead_info_train_filename)

        lateral_dist_to_ego_rl_train_filename = 'lateral_dist_to_ego_rl_{}.npy'.format(i)
        lateral_dist_to_ego_rl_train_file = os.path.join('./converted_dataset/diverse_train', lateral_dist_to_ego_rl_train_filename)

        lateral_dist_to_target_rl_train_filename = 'lateral_dist_to_target_rl_{}.npy'.format(i)
        lateral_dist_to_target_rl_train_file = os.path.join('./converted_dataset/diverse_train', lateral_dist_to_target_rl_train_filename)

        ego_bias_pos_neg_info_train_filename = 'ego_bias_pos_neg_info_{}.npy'.format(i)
        ego_bias_pos_neg_info_train_file = os.path.join('./converted_dataset/diverse_train', ego_bias_pos_neg_info_train_filename)

        target_bias_pos_neg_info_train_filename = 'target_bias_pos_neg_info_{}.npy'.format(i)
        target_bias_pos_neg_info_train_file = os.path.join('./converted_dataset/diverse_train', target_bias_pos_neg_info_train_filename)

        abs_bias_compare_info_train_filename = 'abs_bias_compare_info_{}.npy'.format(i)
        abs_bias_compare_info_train_file = os.path.join('./converted_dataset/diverse_train', abs_bias_compare_info_train_filename)

        target_reference_line_distrib_train_filename = 'target_reference_line_distrib_{}.npy'.format(i)
        target_reference_line_distrib_train_file = os.path.join('./converted_dataset/diverse_train', target_reference_line_distrib_train_filename)

        ego_target_reference_line_distrib_train_filename = 'ego_target_reference_line_distrib_{}.npy'.format(i)
        ego_target_reference_line_distrib_train_file = os.path.join('./converted_dataset/diverse_train', ego_target_reference_line_distrib_train_filename)

        frenet_s_on_reference_line_train_filename = 'frenet_s_on_reference_line_{}.npy'.format(i)
        frenet_s_on_reference_line_train_file = os.path.join('./converted_dataset/diverse_train', frenet_s_on_reference_line_train_filename)

        with open(reference_line_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Reference_Line_Cluster_List_Train[blk_indexes])

        with open(reference_line_mask_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Reference_Line_Mask_Train[blk_indexes])

        with open(planned_path_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Planned_Path_List_Train[blk_indexes])

        with open(navi_drive_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Navi_Drive_List_Train[blk_indexes])

        with open(ego_state_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Ego_State_List_Train[blk_indexes])

        with open(current_context_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Current_Context_List_Train[blk_indexes])

        with open(next_context_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Next_Context_List_Train[blk_indexes])

        with open(lookahead_info_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Lookahead_Info_Cluster_List_Train[blk_indexes])

        with open(lateral_dist_to_ego_rl_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Lateral_Dist_To_Ego_Ref_Line_Train[blk_indexes])

        with open(lateral_dist_to_target_rl_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Lateral_Dist_To_Target_Ref_Line_Train[blk_indexes])

        # ----
        with open(ego_bias_pos_neg_info_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Ego_Bias_Pos_Neg_Info_Train[blk_indexes])

        with open(target_bias_pos_neg_info_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Target_Bias_Pos_Neg_Info_Train[blk_indexes])

        with open(abs_bias_compare_info_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Abs_Bias_Compare_Info_Train[blk_indexes])

        with open(target_reference_line_distrib_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Target_Reference_Line_Distrib_Train[blk_indexes])

        with open(ego_target_reference_line_distrib_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Ego_Target_Reference_Line_Distrib_Train[blk_indexes])

        with open(frenet_s_on_reference_line_train_file, 'wb') as f:
            if i<num_blk-1:
                from_idx = i*avg_num_per_block
                to_idx = (i+1)*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:to_idx]
            else:
                from_idx = i*avg_num_per_block
                blk_indexes = train_data_indexes[from_idx:]
            np.save(f, Frenet_S_On_Reference_Line_Train[blk_indexes])

    Reference_Line_Cluster_List_Train = []
    Reference_Line_Mask_Train = []
    Planned_Path_List_Train = []
    Navi_Drive_List_Train = []
    Ego_State_List_Train = []
    Current_Context_List_Train = []
    Next_Context_List_Train = []
    Lookahead_Info_Cluster_List_Train = []
    Lateral_Dist_To_Ego_Ref_Line_Train = []
    Lateral_Dist_To_Target_Ref_Line_Train = []
    Ego_Bias_Pos_Neg_Info_Train = [] 
    Target_Bias_Pos_Neg_Info_Train = []
    Abs_Bias_Compare_Info_Train = []
    Target_Reference_Line_Distrib_Train = []
    Ego_Target_Reference_Line_Distrib_Train = []
    Frenet_S_On_Reference_Line_Train = []
    print("training part finished.")

    ############
    # save test
    ############
    reference_line_test_file = './converted_dataset/diverse_test/reference_line_0.npy'
    reference_line_mask_test_file = './converted_dataset/diverse_test/reference_line_mask_0.npy'
    planned_path_test_file = './converted_dataset/diverse_test/planned_path_0.npy'
    navi_drive_test_file = './converted_dataset/diverse_test/navi_drive_0.npy'
    ego_state_test_file = './converted_dataset/diverse_test/ego_state_0.npy'
    current_context_test_file = './converted_dataset/diverse_test/current_context_0.npy'
    next_context_test_file = './converted_dataset/diverse_test/next_context_0.npy'
    lookahead_info_test_file = './converted_dataset/diverse_test/lookahead_info_0.npy'
    lateral_dist_to_ego_rl_test_file = './converted_dataset/diverse_test/lateral_dist_to_ego_rl_0.npy'
    lateral_dist_to_target_rl_test_file = './converted_dataset/diverse_test/lateral_dist_to_target_rl_0.npy'
    ego_bias_pos_neg_info_test_file = './converted_dataset/diverse_test/ego_bias_pos_neg_info_0.npy'
    target_bias_pos_neg_info_test_file = './converted_dataset/diverse_test/target_bias_pos_neg_info_0.npy'
    abs_bias_compare_info_test_file = './converted_dataset/diverse_test/abs_bias_compare_info_0.npy'
    target_reference_line_distrib_test_file = './converted_dataset/diverse_test/target_reference_line_distrib_0.npy'
    ego_target_reference_line_distrib_test_file = './converted_dataset/diverse_test/ego_target_reference_line_distrib_0.npy'
    frenet_s_on_reference_line_test_file = './converted_dataset/diverse_test/frenet_s_on_reference_line_0.npy'

    Reference_Line_Cluster_List_Test = np.array(Reference_Line_Cluster_List_Test, dtype=np.float16)
    Reference_Line_Mask_Test = np.array(Reference_Line_Mask_Test, dtype=np.int16)
    Planned_Path_List_Test = np.array(Planned_Path_List_Test, dtype=np.float16)
    Navi_Drive_List_Test = np.array(Navi_Drive_List_Test, dtype=np.int16)
    Ego_State_List_Test = np.array(Ego_State_List_Test, dtype=np.int16)
    Current_Context_List_Test = np.array(Current_Context_List_Test, dtype=np.int16)
    Next_Context_List_Test = np.array(Next_Context_List_Test, dtype=np.int16)
    Lookahead_Info_Cluster_List_Test = np.array(Lookahead_Info_Cluster_List_Test, dtype=np.float16)
    Lateral_Dist_To_Ego_Ref_Line_Test = np.array(Lateral_Dist_To_Ego_Ref_Line_Test, dtype=np.float16)
    Lateral_Dist_To_Target_Ref_Line_Test = np.array(Lateral_Dist_To_Target_Ref_Line_Test, dtype=np.float16)
    #
    Ego_Bias_Pos_Neg_Info_Test = np.array(Ego_Bias_Pos_Neg_Info_Test, dtype=np.int16)
    Target_Bias_Pos_Neg_Info_Test = np.array(Target_Bias_Pos_Neg_Info_Test, dtype=np.int16)
    Abs_Bias_Compare_Info_Test = np.array(Abs_Bias_Compare_Info_Test, dtype=np.int16)
    Target_Reference_Line_Distrib_Test = np.array(Target_Reference_Line_Distrib_Test, dtype=np.int16)
    Ego_Target_Reference_Line_Distrib_Test = np.array(Ego_Target_Reference_Line_Distrib_Test, dtype=np.int16)
    Frenet_S_On_Reference_Line_Test = np.array(Frenet_S_On_Reference_Line_Test, dtype=np.float16)

    with open(reference_line_test_file, 'wb') as f:
        np.save(f, Reference_Line_Cluster_List_Test[test_data_indexes])

    with open(reference_line_mask_test_file, 'wb') as f:
        np.save(f, Reference_Line_Mask_Test[test_data_indexes])

    with open(planned_path_test_file, 'wb') as f:
        np.save(f, Planned_Path_List_Test[test_data_indexes])

    with open(navi_drive_test_file, 'wb') as f:
        np.save(f, Navi_Drive_List_Test[test_data_indexes])

    with open(ego_state_test_file, 'wb') as f:
        np.save(f, Ego_State_List_Test[test_data_indexes])

    with open(current_context_test_file, 'wb') as f:
        np.save(f, Current_Context_List_Test[test_data_indexes])

    with open(next_context_test_file, 'wb') as f:
        np.save(f, Next_Context_List_Test[test_data_indexes])

    with open(lookahead_info_test_file, 'wb') as f:
        np.save(f, Lookahead_Info_Cluster_List_Test[test_data_indexes])

    with open(lateral_dist_to_ego_rl_test_file, 'wb') as f:
        np.save(f, Lateral_Dist_To_Ego_Ref_Line_Test[test_data_indexes])
        
    with open(lateral_dist_to_target_rl_test_file, 'wb') as f:
        np.save(f, Lateral_Dist_To_Target_Ref_Line_Test[test_data_indexes])

    with open(ego_bias_pos_neg_info_test_file, 'wb') as f:
        np.save(f, Ego_Bias_Pos_Neg_Info_Test[test_data_indexes])

    with open(target_bias_pos_neg_info_test_file, 'wb') as f:
        np.save(f, Target_Bias_Pos_Neg_Info_Test[test_data_indexes])

    with open(abs_bias_compare_info_test_file, 'wb') as f:
        np.save(f, Abs_Bias_Compare_Info_Test[test_data_indexes])

    with open(target_reference_line_distrib_test_file, 'wb') as f:
        np.save(f, Target_Reference_Line_Distrib_Test[test_data_indexes])

    with open(ego_target_reference_line_distrib_test_file, 'wb') as f:
        np.save(f, Ego_Target_Reference_Line_Distrib_Test[test_data_indexes])

    with open(frenet_s_on_reference_line_test_file, 'wb') as f:
        np.save(f, Frenet_S_On_Reference_Line_Test[test_data_indexes])

    print("test part finished, done!")


