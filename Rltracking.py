from __future__ import division

import numpy as np
import vtk
import time
from vtk import*

from numpy.testing import (assert_, assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_almost_equal,
                           run_module_suite)

from dipy.data import get_data
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import CsaOdfModel
from dipy.sims.phantom import orbital_phantom
from dipy.core.gradients import gradient_table
from dipy.viz import actor, window, ui
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from numpy.linalg import norm

fimg, fbvals, fbvecs = get_data('small_64D')
bvals = np.load(fbvals)
bvecs = np.load(fbvecs)
bvecs[np.isnan(bvecs)] = 0

gtab = gradient_table(bvals, bvecs)


def f1(t):
    x = np.linspace(-1, 1, len(t))
    y = np.linspace(-1, 1, len(t))
    z = np.zeros(x.shape)
    return x, y, z


def f2(t):
    x = np.linspace(-1, 1, len(t))
    y = -np.linspace(-1, 1, len(t))
    z = np.zeros(x.shape)
    return x, y, z

class Node(object):
    def __init__(self, position, value=0):
        self.position = position
        self.value = value
        self.previous = []
        self.next = []
"""
class State_action(object):
    def __init__(self, direction, position, value=0, index=0):
        self.direction = direction
        self.position = position
        self.value = value
        self.index = index
"""

class Connection(object):
    def __init__(self, index, direction):
        self.index = index
        self.direction = direction

class Seed(object):
    def __init__(self, position, index=0):
        self.position = position
        self.index = index
        self.track1 = []
        self.track2 = []
        self.nodes1 = []
        self.nodes2 = []

class Seed_node_graph(object):
    def __init__(self, graph, value):
        self.graph = graph
        self.value = value

class ReinforcedTracking(object):
    def __init__(self, direction_getter, tissue_classifier, start_points,
                 goal_points, start_radius=3, goal_radius=2,resolution=1,
                 step_size=1, reward_positive=10000, reward_negative=-200,
                 positive=True, alfa=0.02, gamma=0.8,max_cross=None, maxlen=100,
                 minlen=20, fixedstep=True, return_all=True, grouping_size=2,
                 angles=0.5,exp_range=3,resolution_exp=10):
        """Creates streamlines by using reinforcement learning and expanding
           graph.
        Parameters
        ----------
        direction_getter : instance of DirectionGetter
            Used to get directions for fiber tracking.
        tissue_classifier : instance of TissueClassifier
            Identifies endpoints and invalid points to inform tracking.
        seeds : array (N, 3)
            Points to seed the tracking. Seed points should be given in point
            space of the track (see ``affine``).
        affine : array (4, 4)
            Coordinate space for the streamline point with respect to voxel
            indices of input data. This affine can contain scaling, rotational,
            and translational components but should not contain any shearing.
            An identity matrix can be used to generate streamlines in "voxel
            coordinates" as long as isotropic voxels were used to acquire the
            data.
        start_radius : float
            ROI region around the start point
        goal_radius : float
            ROI region around the goal point
        step_size : float
            Step size used for tracking.
        reward_positive: float
            reward for positive feedback in RL learning
        reward_negative: float
            reward for negative feedback in RL learning
        positive: bool
            If Ture, then give positive reward, otherwise give negative reward
        alfa: float
            learning rate
        gamma: float
            discount factor
        max_cross : int or None
            The maximum number of direction to track from each seed in crossing
            voxels. By default all initial directions are tracked.
        maxlen : int
            Maximum number of steps to track from seed. Used to prevent
            infinite loops.
        fixedstep : bool
            If true, a fixed stepsize is used, otherwise a variable step size
            is used.
        return_all : bool
            If true, return all generated streamlines, otherwise only
            streamlines reaching end points or exiting the image.
        grouping_size : float
            Size for clusering a new tracking point into one of the graph
            struture
        """
        self.direction_getter = direction_getter
        self.tissue_classifier = tissue_classifier
        self.shape0 = tissue_classifier.shape[0]
        self.shape1 = tissue_classifier.shape[1]
        self.shape2 = tissue_classifier.shape[2]
        self.start_radius = start_radius
        self.goal_radius = goal_radius
        self.index = 0
        self.indexs = []
        self.resolution_exp = resolution_exp
        self.start_points = np.array(start_points)
        self.start_points_alter = np.array(start_points)
        #self.direc = direc
        self.angles = angles
        self.exp_range = exp_range
        #self.start_point = np.array(start_point)
        self.maxlen = maxlen
        self.minlen = minlen
        self.grouping_size = grouping_size
        if len(self.start_points.shape) == 1:
            self.seed = self.start_points
        else:
            self.seed = self.start_points[0]
        self.goal_points = np.array(goal_points)
        self.goal_points_alter = np.array(goal_points)
        if len(self.goal_points.shape) == 1:
            self.goal_point = self.goal_points
        else:
            self.goal_point = self.goal_points[0]
        #self.goal_point = np.array(goal_point)
        self.resolution = resolution
        self.step_size = step_size
        self.reward_positive = reward_positive
        self.reward_negative = reward_negative
        self.alfa = alfa
        self.gamma = gamma
        self.dir_range = 3
        self.positive = positive
        self.seed_nodes = np.array(self.seed)
        self.seeds_nodes_graph = []
        self.graph = np.array(self.seed)
        self.value = [0.0]
        self.exp_graph_general = np.array(self.seed)
        self.exp_value_general = np.array([0.0])
        self.exp_direc_general = np.array([0,0,0])
        self.exp_graph_alter = self.exp_graph_general
        self.exp_value_alter = self.exp_value_general
        self.exp_direc_alter = self.exp_direc_general
        self.streamlines = []
        streamlines_onetrack,seed_onetrack = self.return_streamline()
        #self.streamlines.append(streamlines_onetrack)
        one_seed_node_graph = Seed_node_graph(self.graph, self.value)
        self.seeds_nodes_graph.append(one_seed_node_graph)
        non_zero = np.where(self.tissue_classifier !=0 )
        n = np.array(non_zero)
        self.n = np.transpose(n)
        self.seeds = []
        self.seeds_nodes_total = []


    def td_learning(self, onetrack1):
        """ Perform TD learning
        """

        if self.positive == True:
            reward = self.reward_positive
        else:
            reward = self.reward_negative

        l1 = len(onetrack1)
        for i in range(l1):
            #!!!replace with tmp = int(onetrack1[l1 - (i +1)])
            if i == 0:
                self.value[int(onetrack1[l1-(i+1)])] = reward
            else:
                # Reinforcement learning (TD)
                self.value[int(onetrack1[l1-(i+1)])] = self.value[int(onetrack1[l1-(i+1)])] + self.alfa*(self.value[int(onetrack1[l1-i])]-self.value[int(onetrack1[l1-(i+1)])])

    def td_learning_general(self, onetrack1):
        """ Perform TD learning
        """

        if self.positive == True:
            reward = self.reward_positive
        else:
            reward = self.reward_negative

        l1 = len(onetrack1)
        for i in range(l1):
            #!!!replace with tmp = int(onetrack1[l1 - (i +1)])
            if i == 0:
                self.exp_value_alter[int(onetrack1[l1-(i+1)])] = reward
            else:
                # Reinforcement learning (TD)
                self.exp_value_alter[int(onetrack1[l1-(i+1)])] = self.exp_value_alter[int(onetrack1[l1-(i+1)])] + self.alfa*(self.exp_value_alter[int(onetrack1[l1-i])]-self.exp_value_alter[int(onetrack1[l1-(i+1)])])

    def find_track_point_general(self,track_point):
        track_point_test = track_point
        if len(self.exp_graph_alter.shape) == 1:
            norm2 = norm(self.exp_graph_alter-track_point_test)
        else:
            norm2 = norm(self.exp_graph_alter-track_point_test,axis=1,ord=2)
        if norm2.min() < self.resolution:
            index_t = np.argmin(norm2)
            return self.exp_direc_alter[index_t]
        else:
            return np.array([0,0,0])


    def find_track_point(self, dirs, track_point):
        track_point_test = track_point + self.step_size * dirs
        if len(self.graph.shape) == 1:
            norm2 = norm(self.graph-track_point_test)
        else:
            norm2 = norm(self.graph-track_point_test,axis=1,ord=2)
        if norm2.min() < self.resolution:
            index_t = np.argmin(norm2)
            return self.value[index_t]
        else:
            return 0.0

    def check_direction(self,t0,t1,t2):
        for i in range(self.dir_range):
            for j in range(self.dir_range):
                for k in range(self.dir_range):
                    if t0+i > self.shape0 or +i == self.shape0:
                        t0 = self.shape0 - i
                    if t1+j > self.shape1 or t1+j == self.shape1:
                        t1 = self.shape1 - j
                    if t2+k > self.shape2 or t2+k == self.shape2:
                        t2 = self.shape2 - k
                    if t0-i < 0:
                        t0 = i
                    if t1 - j<0:
                        t1 = j
                    if t2 - k <0:
                        t2 = k
                    dir_sub = self.direction_getter[t0+i, t1+j, t2+k, 0,:]
                    if not dir_sub.all() == False:
                        t0 = t0 + i
                        t1 = t1 + j
                        t2 = t2 + k
                        return t0,t1,t2
                    dir_sub = self.direction_getter[t0-i,t1-j,t2-k,0,:]
                    if not dir_sub.all() == False:
                        t0 = t0 - i
                        t1 = t1 - j
                        t2 = t2 - k
                        return t0,t1,t2
        return -1,-1,-1

    def show_graph_values(self, show_final=True,show_node=True,show_start=True,show_node_general=True):

        #streamlines_actor = actor.line(streamlines)
        time_count = 0
        renderwindow = vtk.vtkRenderWindow()
        renderwindow.SetSize(1000,1000)
        r = window.Renderer()
        #r = vtk.vtkRenderer()
        #r.clear()
        r.background((1, 1, 1))
        #r.SetBackground(1,1,1)
        renderwindow.AddRenderer(r)
        renderwindowInteractor = vtkRenderWindowInteractor()
        #r = vtk.vtkRenderWindowInteractor()
        renderwindowInteractor.SetRenderWindow(renderwindow)
        camera = vtkCamera()
        camera.SetPosition(-50,-50,-50)
        camera.SetFocalPoint(0,0,0)
        #r.SetActiveCamera(camera)
        streamlines_sub = []
        #r.add(streamlines_actor)
        #actor_slicer = actor.slicer(FA, interpolation='nearest')
        #r.add(actor_slicer)
        if not show_final:
            window.show(r)

        colors2 = np.array((0,1,1))
        color3 = np.array((1,1,1))
        colors = np.array((1,0,0))
        colors1 = np.array((0,1,0))
        #goal_point = np.array([35,67,41])
        #goal_point = np.array([37,53,59])
        #starting_point = np.array([53,65,40])
        #starting_point = np.array([42,55,20])
        #goal_point = goal_point[None,:]
        #point_actor2 = fvtk.point(self.seed[None,:],colors2, point_radius=0.5)#point_radius=self.start_radius)
        point_actor2 = fvtk.point(self.start_points,colors2, point_radius=0.5)
        #point_actor1 = fvtk.point(self.goal_points[None,:],colors1, point_radius=self.goal_radius)
        point_actor1 = fvtk.point(self.goal_points,colors1, point_radius=0.5)
        if show_start:
            r.add(point_actor1)
        #r.AddActor(point_actor1)
            r.add(point_actor2)
        #r.AddActor(point_actor2)

        #def time_event(obj, ev):
            #time.sleep(20)
        """Alter
        """
        if len(self.exp_graph_general.shape) == 1:
            #colors = np.random.rand(1,3)
            colors = np.array((1,1,1))
            if np.array(self.exp_value_general) > 0:
                colors = np.array((1,1 - self.exp_value_general[0]/100,1 - self.exp_value_general[0]/100))

        else:
            max_value = np.abs(self.exp_value_general.max())
            ss = np.where(self.exp_value_general<0)[0]
            colors = np.ones((self.exp_graph_general.shape[0],3))
            #colors[:,2] = 1 - seeds_nodes_graph[i].value/max_value
            #colors[:,1] = 1 - seeds_nodes_graph[i].value/max_value
            colors[ss,2] = 1
            colors[ss,1] = 1

            sss = np.where(self.exp_value_general>0)[0]
            colors[sss,1] = 0
            colors[sss,2] = 0

        if len(self.exp_graph_general.shape) == 1:
            point_actor = fvtk.point(self.exp_graph_general[None,:],colors, point_radius=0.3)
        else:
            point_actor = fvtk.point(self.exp_graph_general,colors, point_radius=0.3)
        if show_node_general:
            r.add(point_actor)

        for i in range(len(self.seeds_nodes_graph)):
            #print(i)
            #r.clear()
            #node_actor = actor.streamtube([nodes[100]])
            if len(self.seeds_nodes_graph[i].graph.shape) == 1:
                #colors = np.random.rand(1,3)
                colors = np.array((1,1,1))
                if np.array(self.seeds_nodes_graph[i].value) > 0:
                    colors = np.array((1,1 - self.seeds_nodes_graph[i].value[0]/100,1 - self.seeds_nodes_graph[i].value[0]/100))

            else:
                max_value = np.abs(self.seeds_nodes_graph[i].value.max())
                ss = np.where(self.seeds_nodes_graph[i].value<0)[0]
                colors = np.ones((self.seeds_nodes_graph[i].graph.shape[0],3))
                #colors[:,2] = 1 - seeds_nodes_graph[i].value/max_value
                #colors[:,1] = 1 - seeds_nodes_graph[i].value/max_value
                colors[ss,2] = 1
                colors[ss,1] = 1

                sss = np.where(self.seeds_nodes_graph[i].value>0)[0]
                colors[sss,1] = 0
                colors[sss,2] = 0

                if self.seeds_nodes_graph[i].value.max()>0:
                    #colors = np.zeros((seeds_nodes_graph[i].graph.shape[0],3))
                    #colors[:,0] = 1
                    streamlines_sub.append(self.seeds_nodes_graph[i].graph)



            if len(self.seeds_nodes_graph[i].graph.shape) == 1:
                point_actor = fvtk.point(self.seeds_nodes_graph[i].graph[None,:],colors, point_radius=0.3)
            else:
                point_actor = fvtk.point(self.seeds_nodes_graph[i].graph,colors, point_radius=0.3)
            if show_node:
                r.add(point_actor)
            #r.AddActor(point_actor)
            #iren = obj
            #iren.GetRenderWindow().Render()
        if not show_final:
            window.show(r)
        if not len(self.streamlines) == 0:
            peak_slicer = actor.line(self.streamlines)
            r.add(peak_slicer)
        if show_final:
            window.show(r)

    def return_streamline(self):
        """ returns one streamline, generates graph and does RL
        """
        decision1 = 0
        decision2 = 0
        decision = 0
        streamline = self.seed
        track_point = self.seed
        node_onetrack = []
        decision1 = 1
        decision2 = 1
        if len(self.graph.shape) == 1:
            index_c = 0
            node_onetrack = self.seed
        if len(self.graph.shape) != 1:
            norm2 = norm(self.graph-self.seed,axis=1,ord=2)
            if norm2.min() < self.resolution:
                index_c = np.argmin(norm2)
                node_onetrack = self.graph[index_c]
            else:
                index_c = self.graph.shape[0]
                self.graph = np.vstack((self.graph,self.seed))
                self.value = np.append(self.value,0.0)
                #node_onetrack = seed
        """Alter
        """
        if len(self.exp_graph_alter.shape) == 1:
            norm_alter = norm(self.exp_graph_alter-self.seed)
            if norm_alter.min() < self.resolution:
                index_alter = np.argmin(norm_alter)
            else:
                index_alter = self.exp_graph_alter.shape[0]
                self.exp_graph_alter = np.vstack((self.exp_graph_alter,self.seed))
                self.exp_value_alter = np.append(self.exp_value_alter,0.0)
                self.exp_direc_alter = np.vstack((self.exp_direc_alter,np.array([0,0,0])))
        if len(self.exp_graph_alter.shape) != 1:
            norm_alter = norm(self.exp_graph_alter-self.seed,axis=1,ord=2)
            if norm_alter.min() < self.resolution:
                index_alter = np.argmin(norm_alter)
                node_onetrack_alter = self.exp_graph_alter[index_alter]
            else:
                index_alter = self.exp_graph_alter.shape[0]
                self.exp_graph_alter = np.vstack((self.exp_graph_alter,self.seed))
                self.exp_value_alter = np.append(self.exp_value_alter,0.0)
                self.exp_direc_alter = np.vstack((self.exp_direc_alter,np.array([0,0,0])))

        seed_onetrack = Seed(self.seed, index_c)
        seed_onetrack.track1 = np.append(seed_onetrack.track1, index_c)
        """Alter
        """
        seed_onetrack_alter = Seed(self.seed, index_alter)
        seed_onetrack_alter.track1 = np.append(seed_onetrack_alter.track1, index_alter)
        """
        if len(self.graph.shape) == 1:
            seed_onetrack.nodes1 = self.graph
        else:
            seed_onetrack.nodes1 = self.graph[index_c]
        """

        def itp(track_point):
            t0 = int(np.round(track_point[0]))
            t1 = int(np.round(track_point[1]))
            t2 = int(np.round(track_point[2]))
            return t0, t1, t2

        t0_init,t1_init,t2_init = itp(track_point)
        """
        if self.direc == True:
            dir_old = -self.direction_getter[t0, t1, t2, 0,:]
        if self.direc == False:
            dir_old = self.direction_getter[t0, t1, t2, 0,:]
        """
        """First initial start direction
        """
        for kk in range(2):
            if kk%2 == 0:
                dir_old = self.direction_getter[t0_init, t1_init, t2_init,0,:]#,int(kk/2) ,:]
            else:
                dir_old = -self.direction_getter[t0_init, t1_init, t2_init,0,:]# int(np.floor(kk/2)),:]
            t0 = t0_init
            t1 = t1_init
            t2 = t2_init
            while(self.tissue_classifier[t0,t1,t2] != 0 ):
                decision1 = 0
                decision2 = 0
                value_single = -500
                t0, t1, t2 = itp(track_point)
                dir_sub = self.direction_getter[t0, t1, t2, 0,:]
                #dir_final = self.direction_getter[t0,t1,t2,0,:]
                if dir_sub.all() == False:
                    t0, t1, t2 = self.check_direction(t0,t1,t2)
                if t0 == -1 and t1 == -1 and t2 == -1:
                    break
                """First direction
                """
                for i in range(5):
                    dir_sub = self.direction_getter[t0, t1, t2, i,:]
                    if dir_sub.all() == True:
                        if np.dot(dir_old,dir_sub)<self.angles:
                                #dir_sub = -dir_sub
                            continue
                        value_single_test = self.find_track_point(dir_sub, track_point)
                        #if value_single_test < self.reward_negative/25:
                        #    continue
                        decision1 = 1
                        if value_single_test > value_single:
                            index_inside = i
                            value_single = value_single_test
                            dir_final = dir_sub
                        """Alter

                        value_single_test_alter = self.find_track_point_general(dir_sub, track_point)
                        if value_single_test_alter > 0:
                            if value_single_test < 0:
                                continue
                            index_inside = i
                            value_single = value_single_test
                            dir_final = dir_sub
                        """
                """
                second direction
                """
                for i in range(5):
                    dir_sub = -self.direction_getter[t0, t1, t2, i,:]
                    if dir_sub.all() == True:
                        if np.dot(dir_old,dir_sub)<self.angles:
                                #dir_sub = -dir_sub
                            continue
                        value_single_test = self.find_track_point(dir_sub, track_point)
                        #if value_single_test < self.reward_negative/25:
                        #    continue
                        decision2 = 1
                        if value_single_test > value_single:
                            index_inside = i
                            value_single = value_single_test
                            dir_final = dir_sub
                        """Alter

                        value_single_test_alter = self.find_track_point_general(dir_sub, track_point)
                        if value_single_test_alter > value_single:
                            if value_single_test < 0:
                                continue
                            index_inside = i
                            value_single = value_single_test_alter
                            dir_final = dir_sub
                        """

                dir_learned = self.find_track_point_general(track_point)
                if np.any(dir_learned):
                    if np.dot(dir_final,dir_learned) > self.angles:
                        #print("im in corporating dir")
                        dir_final = (0.3*dir_learned+0.3*dir_old+0.7*dir_final)/norm(0.3*dir_learned+0.3*dir_old+0.7*dir_final)


                if decision1 == 0 and decision2 == 0:
                    break

                #dir_old = dir_final
                #track_point = track_point + self.step_size * dir_final
                """Adding computing direction
                """
                if len(self.exp_graph_alter.shape) == 1:
                    norm_final = norm(self.exp_graph_alter-track_point)
                else:
                    norm_final = norm(self.exp_graph_alter-track_point,axis=1,ord=2)
                if norm_final.min() < self.resolution_exp:
                    """
                    if np.sum(norm_final < self.resolution) < self.exp_range:
                        index_tt = np.argmin(norm_final)
                        node_near = self.exp_graph_alter[index_tt]
                        dir_tt = self.exp_direc_alter[index_tt]
                        if not norm(node_near-track_point) == 0:
                            dir_t = (node_near-track_point)/norm(node_near-track_point)
                            if np.dot(dir_old,dir_t)>self.angles:
                                #print("im here inference")
                                if np.dot(dir_old,dir_tt)<0:
                                    dir_final = (0.2*dir_old+0.2*dir_final+dir_t-0.1*dir_tt)/norm(0.2*dir_old+0.2*dir_final+dir_t-0.1*dir_tt)
                                else:
                                    dir_final = (0.2*dir_old+0.2*dir_final+dir_t+0.1*dir_tt)/norm(0.2*dir_old+0.2*dir_final+dir_t+0.1*dir_tt)
                    """
                    if np.sum(norm_final < self.resolution) > self.exp_range or np.sum(norm_final < self.resolution) == self.exp_range:
                        #print("im here")
                        index_tt = np.argmin(norm_final)
                        #index_tt = np.where(norm_final<self.resolution)
                        node_near = self.exp_graph_alter[index_tt]
                        dir_t = self.exp_direc_alter[index_tt]
                        #dir_t = np.sum(self.exp_direc_alter[index_tt],axis=0)/norm(self.exp_direc_alter[index_tt],axis=0)
                        if np.any(dir_t) and np.dot(dir_old,dir_t)>self.angles:
                            print("im here")
                            dir_final = (0.3*dir_old+dir_final+0.5*dir_t)/norm(0.3*dir_old+dir_final+0.5*dir_t)
                        if np.any(dir_t) and np.dot(dir_old,dir_t)<0:
                            print("im here")
                            dir_final = (0.3*dir_old+dir_final-0.5*dir_t)/norm(0.3*dir_old+dir_final-0.5*dir_t)
                        """
                        if not np.any(dir_t):
                            index_tt = np.argmin(norm_final)
                            node_near = self.exp_graph_alter[index_tt]
                            dir_tt = self.exp_direc_alter[index_tt]
                            if not norm(node_near-track_point) == 0:
                                dir_t = (node_near-track_point)/norm(node_near-track_point)
                                if np.dot(dir_old,dir_t)>self.angles:
                                    #print("im here inference")
                                    if np.dot(dir_old,dir_tt)<0:
                                        dir_final = (0.2*dir_old+0.2*dir_final+dir_t-0.1*dir_tt)/norm(0.2*dir_old+0.2*dir_final+dir_t-0.1*dir_tt)
                                    else:
                                        dir_final = (0.2*dir_old+0.2*dir_final+dir_t+0.1*dir_tt)/norm(0.2*dir_old+0.2*dir_final+dir_t+0.1*dir_tt)

                        """
                dir_old = dir_final
                track_point = track_point + self.step_size * dir_final

                if len(self.graph.shape) == 1:
                    norm2 = norm(self.graph-track_point)
                else:
                    norm2 = norm(self.graph-track_point,axis=1,ord=2)
                """Alter
                """
                if len(self.exp_graph_alter.shape) == 1:
                    norm_alter = norm(self.exp_graph_alter-track_point)
                else:
                    norm_alter = norm(self.exp_graph_alter-track_point,axis=1,ord=2)

                if norm2.min() < self.resolution:
                    index_t = np.argmin(norm2)
                    if not np.any(seed_onetrack.track1 == index_t):
                        seed_onetrack.track1 = np.append(seed_onetrack.track1,index_t)
                        if len(self.graph.shape) == 1:
                            seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, self.graph))
                        else:
                            seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, self.graph[int(index_t)]))
                else:
                    if len(self.graph.shape) == 1:
                        index_t = 1
                    else:
                        index_t = self.graph.shape[0]
                    self.graph = np.vstack((self.graph,track_point))
                    self.value = np.append(self.value,0.0)
                    seed_onetrack.track1 = np.append(seed_onetrack.track1, index_t)
                    if len(self.graph.shape) == 1:
                        seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, self.graph))
                    else:
                        seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, self.graph[int(index_t)]))

                """Alter
                """
                if norm_alter.min() < self.resolution:
                    index_alter = np.argmin(norm_alter)
                    if not np.any(seed_onetrack_alter.track1 == index_alter):
                        seed_onetrack_alter.track1 = np.append(seed_onetrack_alter.track1,index_alter)
                else:
                    if len(self.exp_graph_alter.shape) == 1:
                        index_alter = 1
                    else:
                        index_alter = self.exp_graph_alter.shape[0]
                    self.exp_direc_alter = np.vstack((self.exp_direc_alter,dir_final))
                    self.exp_graph_alter = np.vstack((self.exp_graph_alter,track_point))
                    self.exp_value_alter = np.append(self.exp_value_alter,0.0)
                    seed_onetrack_alter.track1 = np.append(seed_onetrack_alter.track1, index_alter)
                streamline = np.vstack((streamline,track_point))
                t0, t1, t2 = itp(track_point)

                if t0 > self.shape0 or t0 == self.shape0:
                    t0 = self.shape0 - 1
                if t1 > self.shape1 or t1 == self.shape1:
                    t1 = self.shape1 - 1
                if t2 > self.shape2 or t2 == self.shape2:
                    t2 = self.shape2 - 1

                #dir_sub = self.direction_getter[t0, t1, t2, 0,:]
            #if dir_sub.all() == False:
            #    t0, t1, t2 = self.check_direction(t0,t1,t2)
        """
        if len(seed_onetrack.nodes1.shape) == 1:
            norm3_track1 = norm(seed_onetrack.nodes1 - self.goal_point)
        else:
            norm3_track1 = norm(seed_onetrack.nodes1 - self.goal_point,axis=1,ord=2)
        """
        if len(streamline.shape) == 1:
            norm3_track1 = norm(streamline - self.goal_points)
        else:
            for i in range(streamline.shape[0]):
                norm3_track1 = norm(streamline[i] - self.goal_points,axis=1,ord=2)
                if norm3_track1.min()<self.goal_radius:
                    self.positive=True
                    #self.streamlines.append(streamline)
                    decision = 1
                    break
            if decision == 0:
                self.positive=False
            #if seed_onetrack.track1.shape[0] > self.maxlen:
            if streamline.shape[0] > self.maxlen:
                self.positive = False
            if self.positive == True:
                self.streamlines.append(streamline)
                self.td_learning_general(seed_onetrack_alter.track1)
                self.exp_graph_general = self.exp_graph_alter
                self.exp_value_general = self.exp_value_alter
                self.exp_direc_general = self.exp_direc_alter
                self.indexs = np.append(self.indexs,self.index)
            else:
                self.exp_graph_alter = self.exp_graph_general
                self.exp_value_alter = self.exp_value_general
                self.exp_direc_alter = self.exp_direc_general
            self.td_learning(seed_onetrack.track1)
        return streamline, seed_onetrack

    def generate_streamlines(self):
        """Generate streamlines connecting two ROI regions
        """
        #non_zero = np.where(self.tissue_classifier !=0 )
        #n = np.array(non_zero)
        #n = np.transpose(n)
        #self.goal_point = self.goal_points[0]
        #self.seeds = []
        if len(self.start_points.shape) == 1:
            self.seeds = []
            self.seeds.append(self.start_points)
            index =  np.argmin(norm(self.n-self.start_points,axis=1,ord=2))
            self.seeds.append(self.n[index])
            for j in range(len(self.seeds)):
            #self.seed = self.start_points
                self.seed = self.seeds[j]
                self.goal_point = self.goal_points
                """
                for k in range(len(self.streamlines)):
                    norm_streamline = norm(self.seed-self.streamlines[k][0])
                    if norm_streamline.min() == 0:
                        return
                """
                if len(self.seed_nodes.shape) == 1:
                    norm2 = norm(self.seed_nodes-self.seed)
                if len(self.seed_nodes.shape) != 1:
                    norm2 = norm(self.seed_nodes-self.seed,axis=1,ord=2)
                if norm2.min() < self.grouping_size:
                    index_c = np.argmin(norm2)
                    self.graph = self.seeds_nodes_graph[index_c].graph
                    self.value = self.seeds_nodes_graph[index_c].value
                    streamlines_onetrack, seed_onetrack = self.return_streamline()
                    self.seeds_nodes_graph[index_c].graph = self.graph
                    self.seeds_nodes_graph[index_c].value = self.value
                else:
                    index_c = self.seed_nodes.shape[0]
                    self.seed_nodes = np.vstack((self.seed_nodes,self.seed))
                    self.graph = self.seed
                    self.value = [0.0]

                    streamlines_onetrack, seed_onetrack = self.return_streamline()
                    one_seed_node_graph = Seed_node_graph(self.graph, self.value)
                    self.seeds_nodes_graph.append(one_seed_node_graph)
        else:
            for i in range(self.start_points.shape[0]):
                self.seeds = []
                self.index = i
                print(i)
                #self.goal_point = self.goal_points[i]
                self.seeds.append(self.start_points[i])
                index =  np.argmin(norm(self.n-self.start_points[i],axis=1,ord=2))
                self.seeds.append(self.n[index])
                """
                for k in range(len(self.streamlines)):
                    norm_streamline = norm(self.seed-self.streamlines[k][0])
                    if norm_streamline.min() == 0:
                        continue
                """
                for j in range(len(self.seeds)):
                #if norm(self.seed - self.start_point) < self.start_radius:
                    self.seed = self.seeds[j]
                    if len(self.seed_nodes.shape) == 1:
                        norm2 = norm(self.seed_nodes-self.seed)
                    if len(self.seed_nodes.shape) != 1:
                        norm2 = norm(self.seed_nodes-self.seed,axis=1,ord=2)
                    if norm2.min() < self.grouping_size:
                        index_c = np.argmin(norm2)
                        self.graph = self.seeds_nodes_graph[index_c].graph
                        self.value = self.seeds_nodes_graph[index_c].value
                        streamlines_onetrack, seed_onetrack = self.return_streamline()
                        self.seeds_nodes_graph[index_c].graph = self.graph
                        self.seeds_nodes_graph[index_c].value = self.value
                    else:
                        index_c = self.seed_nodes.shape[0]
                        self.seed_nodes = np.vstack((self.seed_nodes,self.seed))
                        self.graph = self.seed
                        self.value = [0.0]

                        streamlines_onetrack, seed_onetrack = self.return_streamline()
                        one_seed_node_graph = Seed_node_graph(self.graph, self.value)
                        self.seeds_nodes_graph.append(one_seed_node_graph)

        self.start_points = np.delete(self.start_points,self.indexs,axis=0)
        self.indexs = []

                #if decision1 == 1:
                #if not streamlines_onetrack == []:
                #    self.streamlines.append(streamlines_onetrack)






def show_mosaic(data):

    renderer = window.Renderer()
    renderer.background((0.5, 0.5, 0.5))

    slice_actor = actor.slicer(data)

    show_m = window.ShowManager(renderer, size=(1200, 900))
    show_m.initialize()

    label_position = ui.TextBlock2D(text='Position:')
    label_value = ui.TextBlock2D(text='Value:')

    result_position = ui.TextBlock2D(text='')
    result_value = ui.TextBlock2D(text='')

    panel_picking = ui.Panel2D(center=(200, 120),
                               size=(250, 125),
                               color=(0, 0, 0),
                               opacity=0.75,
                               align="left")

    panel_picking.add_element(label_position, 'relative', (0.1, 0.55))
    panel_picking.add_element(label_value, 'relative', (0.1, 0.25))

    panel_picking.add_element(result_position, 'relative', (0.45, 0.55))
    panel_picking.add_element(result_value, 'relative', (0.45, 0.25))

    show_m.ren.add(panel_picking)

    """
    Add a left-click callback to the slicer. Also disable interpolation so you can
    see what you are picking.
    """

    renderer.clear()
    renderer.projection('parallel')

    result_position.message = ''
    result_value.message = ''

    show_m_mosaic = window.ShowManager(renderer, size=(1200, 900))
    show_m_mosaic.initialize()


    def left_click_callback_mosaic(obj, ev):
        """Get the value of the clicked voxel and show it in the panel."""
        event_pos = show_m_mosaic.iren.GetEventPosition()

        obj.picker.Pick(event_pos[0],
                        event_pos[1],
                        0,
                        show_m_mosaic.ren)

        i, j, k = obj.picker.GetPointIJK()
        result_position.message = '({}, {}, {})'.format(str(i), str(j), str(k))
        result_value.message = '%.8f' % data[i, j, k]



    cnt = 0

    X, Y, Z = slice_actor.shape[:3]

    rows = 10
    cols = 15
    border = 10

    for j in range(rows):
        for i in range(cols):
            slice_mosaic = slice_actor.copy()
            slice_mosaic.display(None, None, cnt)
            slice_mosaic.SetPosition((X + border) * i,
                                     0.5 * cols * (Y + border) - (Y + border) * j,
                                     0)
            slice_mosaic.SetInterpolate(False)
            slice_mosaic.AddObserver('LeftButtonPressEvent',
                                     left_click_callback_mosaic,
                                     1.0)
            renderer.add(slice_mosaic)
            cnt += 1
            if cnt > Z:
                break
        if cnt > Z:
            break

    renderer.reset_camera()
    renderer.zoom(1.6)

    show_m_mosaic.ren.add(panel_picking)
    show_m_mosaic.start()


def build_phantom(fname):

    N = 200
    S0 = 100
    N_angles = 32
    N_radii = 20
    vol_shape = (100, 100, 100)
    origin = (50, 50, 50)
    scale = (30, 30, 30)
    t = np.linspace(0, 2 * np.pi, N)
    angles = np.linspace(0, 2 * np.pi, N_angles)
    radii = np.linspace(0.2, 2, N_radii)

    vol1 = orbital_phantom(gtab,
                           func=f1,
                           t=t,
                           datashape=vol_shape + (len(bvals),),
                           origin=origin,
                           scale=scale,
                           angles=angles,
                           radii=radii,
                           S0=S0)

    vol2 = orbital_phantom(gtab,
                           func=f2,
                           t=t,
                           datashape=vol_shape + (len(bvals),),
                           origin=origin,
                           scale=scale,
                           angles=angles,
                           radii=radii,
                           S0=S0)

    vol = vol1 + vol2

    np.save(fname, vol)


def show_graph_values(rt, bundle, show_final=True):

    #r.add(streamlines_actor)
    #actor_slicer = actor.slicer(FA, interpolation='nearest')
    #r.add(actor_slicer)
    if not show_final:
        window.show(r)

    colors2 = np.array((0,1,1))
    color3 = np.array((1,1,1))
    colors = np.array((1,0,0))
    colors1 = np.array((0,1,0))
    goal_point = np.array([35,67,41])
    #goal_point = np.array([37,53,59])
    starting_point = np.array([53,65,40])
    #starting_point = np.array([42,55,20])
    goal_point = goal_point[None,:]
    #r.add(point_actor1)
    #r.AddActor(point_actor1)
    #r.add(point_actor2)
    #r.AddActor(point_actor2)
    r = window.Renderer()
    r.add(actor.line(bundle))

    #def time_event(obj, ev):
        #time.sleep(20)
    for i in range(len(rt.seeds_nodes_graph)):
        #print(i)
        #r.clear()
        #node_actor = actor.streamtube([nodes[100]])
        if len(rt.seeds_nodes_graph[i].graph.shape) == 1:
            #colors = np.random.rand(1,3)
            colors = np.array((1,1,1))
            if np.array(rt.seeds_nodes_graph[i].value) > 0:
                colors = np.array((1,1 - rt.seeds_nodes_graph[i].value[0]/100,1 - rt.seeds_nodes_graph[i].value[0]/100))

        else:
            max_value = np.abs(rt.seeds_nodes_graph[i].value.max())
            ss = np.where(rt.seeds_nodes_graph[i].value<0)[0]
            colors = np.ones((rt.seeds_nodes_graph[i].graph.shape[0],3))
            #colors[:,2] = 1 - seeds_nodes_graph[i].value/max_value
            #colors[:,1] = 1 - seeds_nodes_graph[i].value/max_value
            colors[ss,2] = 1
            colors[ss,1] = 1

            sss = np.where(rt.seeds_nodes_graph[i].value>0)[0]
            colors[sss,1] = 0
            colors[sss,2] = 0

            #if rt.seeds_nodes_graph[i].value.max()>0:
                #colors = np.zeros((seeds_nodes_graph[i].graph.shape[0],3))
                #colors[:,0] = 1
                #streamlines_sub.append(rt.seeds_nodes_graph[i].graph)



        if len(rt.seeds_nodes_graph[i].graph.shape) == 1:
            point_actor = fvtk.point(rt.seeds_nodes_graph[i].graph[None,:],colors, point_radius=0.3)
        else:
            point_actor = fvtk.point(rt.seeds_nodes_graph[i].graph,colors, point_radius=0.3)
        r.add(point_actor)
        #r.AddActor(point_actor)
        #iren = obj
        #iren.GetRenderWindow().Render()
    if not show_final:
        window.show(r)
    #peak_slicer = actor.line(streamlines_sub)
    #r.add(peak_slicer)
    if show_final:
        window.show(r)
        #renderwindow.Render()
        #renderwindowInteractor.Initialize()
        #renderwindowInteractor.AddObserver('TimerEvent', time_event)
        #timerId = renderwindowInteractor.CreateRepeatingTimer(100)
        #renderwindowInteractor.Start()

if __name__ == "__main__":

    """
    vol = np.load('vol_complex_phantom2.npy')
    tensor_model = TensorModel(gtab)
    tensor_fit = tensor_model.fit(vol)
    FA = tensor_fit.fa
    # print vol
    FA[np.isnan(FA)] = 0
    # 686 -> expected FA given diffusivities of [1500, 400, 400]
    l1, l2, l3 = 1500e-6, 400e-6, 400e-6
    expected_fa = (np.sqrt(0.5) *
                   np.sqrt((l1 - l2)**2 + (l2-l3)**2 + (l3-l1)**2) /
                   np.sqrt(l1**2 + l2**2 + l3**2))
    mask = FA > 0.1
    csa_model = CsaOdfModel(gtab, 8)
    from dipy.direction import peaks_from_model
    from dipy.data import get_sphere
    from numpy.linalg import norm
    pam = peaks_from_model(csa_model, vol, get_sphere('repulsion724'),
                           relative_peak_threshold=.5,
                           min_separation_angle=25,
                           mask=mask,
                           parallel=True)
    """
#    seeds = utils.seeds_from_mask(seed_mask, density=[2, 2, 2], affine=affine)
    dname = 'C:\\Users\Tingyi\\Dropbox\\Tingyi\\ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2\\ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2\\'

    fbvals = dname +  'NoArtifacts_Relaxation.bvals'
    fbvecs = dname + 'NoArtifacts_Relaxation.bvecs'
    fdwi = dname + 'NoArtifacts_Relaxation.nii.gz'

    from dipy.io.image import load_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.segment.mask import median_otsu

    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    data, affine = load_nifti(fdwi)

    maskdata, mask = median_otsu(data, 3, 1, False, vol_idx=range(1, 20))

    from dipy.reconst.dti import TensorModel

    tensor_model = TensorModel(gtab, fit_method='WLS')
    tensor_fit = tensor_model.fit(data, mask)

    FA = tensor_fit.fa

    from dipy.reconst.shm import CsaOdfModel
    from dipy.data import get_sphere
    from dipy.direction.peaks import peaks_from_model

    sphere = get_sphere('repulsion724')

    csamodel = CsaOdfModel(gtab, 4)
    pam = peaks_from_model(model=csamodel,
                                data=maskdata,
                                sphere=sphere,
                                relative_peak_threshold=.5,
                                min_separation_angle=25,
                                mask=mask,
                                return_odf=False,
                                normalize_peaks=True)
