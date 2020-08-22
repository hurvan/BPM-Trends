# -*- coding: utf-8 -*-
"""QT graph widget to display a BPM 2D spectrogram"""

# Imports
import sys
import numpy as np
from datetime import datetime, timedelta
import scipy.io
import resource
import matplotlib.pyplot as plt
import re
from functools import partial
import os

# QT imports
import pyqtgraph as pg
from PyQt4 import QtCore, QtGui

# Tango imports
import PyTango as PT


"""Global initial variables"""
minutes = 16
updateFrequency = 2   # Hz

"""Color styles for the Qbuttons"""
activeStyle = ("QPushButton {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(0, 255, 0, 175), stop:1 rgba(0, 150, 0, 255));"
                  "border-radius: 5px;"
                  "color: black;"
                  "min-height: 30px;"
                  "max-height: 30px;"
                  "border: 0.5px solid lightgray;}"
                  ".QPushButton:Hover {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(0, 255, 0, 165), stop:1 rgba(0, 175, 0, 225));}"
                  ".QPushButton:pressed {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(0, 150, 0, 255), stop:1 rgba(0, 255, 0, 225));}"
                  ".QPushButton:checked {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(0, 80, 0, 255), stop:1 rgba(0, 255, 0, 255));}")
stopStyle = ("QPushButton {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(255, 0, 0, 175), stop:1 rgba(150, 0, 0, 255));"
                  "border-radius: 5px;"
                  "color: white;"
                  "min-height: 30px;"
                  "max-height: 30px;"
                  "border: 0.5px solid lightgray;}"
                  ".QPushButton:Hover {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(255, 0, 0, 165), stop:1 rgba(175, 0, 0, 225));}"
                  ".QPushButton:pressed {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(150, 0, 0, 255), stop:1 rgba(255, 0, 0, 225));}"
                  ".QPushButton:checked {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(80, 0, 0, 255), stop:1 rgba(255, 0, 0, 255));}")
bpmStyle = ("QPushButton {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(255, 255, 0, 125), stop:1 rgba(255, 255, 0, 175));"
                  "border-radius: 5px;"
                  "color: black;"
                  "min-height: 30px;"
                  "max-height: 30px;"
                  "border: 0.5px solid lightgray;}"
                  ".QPushButton:Hover {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(255, 255, 0, 165), stop:1 rgba(175, 175, 0, 225));}"
                  ".QPushButton:pressed {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(150, 150, 0, 255), stop:1 rgba(255, 255, 0, 225));}"
                  ".QPushButton:checked {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(80, 80, 0, 255), stop:1 rgba(255, 255, 0, 255));}")
corStyle = ("QPushButton {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(0, 0, 255, 125), stop:1 rgba(0, 0, 150, 175));"
                  "border-radius: 5px;"
                  "color: white;"
                  "min-height: 30px;"
                  "max-height: 30px;"
                  "border: 0.5px solid lightgray;}"
                  ".QPushButton:Hover {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(0, 0, 255, 165), stop:1 rgba(0, 0, 175, 225));}"
                  ".QPushButton:pressed {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(0, 0, 150, 255), stop:1 rgba(0, 0, 255, 225));}"
                  ".QPushButton:checked {"
                  "background: qlineargradient(spread:pad, x1:0 y1:0, x2:0 y2:1, "
                  "stop:0 rgba(0, 0, 80, 255), stop:1 rgba(0, 0, 255, 255));}")

class DataReader():
    def __init__(self, signal, ring, axis):
        self.signal = signal
        if ring.lower() == 'r1':
            self.deviceX = PT.DeviceProxy('r1/ctl/sofb-01')    
            self.deviceY = PT.DeviceProxy('r1/ctl/sofb-02')
            self.scaleFactor = 0.001 # 2019-08-12 - scale factor updated to same as for r3 after summer BPM update. Was 1000 before. 
        elif ring.lower() == 'r3':
            self.deviceX = PT.DeviceProxy('r3/ctl/sofb-01')    
            self.deviceY = PT.DeviceProxy('r3/ctl/sofb-02')   
            self.scaleFactor = 0.001
        self.axis = axis

#        sensorNames = list(self.deviceX.sensor_names) + list(self.deviceY.sensor_names)
        self.xDevsSize = len(self.deviceX.sensor_names)
        self.yDevsSize = len(self.deviceY.sensor_names)
        
#        for name in sensorNames:
#            if 'XPos' in name:
#                self.xDevsSize += 1
#            elif 'YPos' in name:
#                self.yDevsSize += 1

    def read(self):
        try:
            if self.axis == 'x':
                sensorVals = self.deviceX.sensor_current_values
            elif self.axis == 'y':
                sensorVals = self.deviceY.sensor_current_values
        except Exception:
            sensorVals = np.zeros(self.xDevsSize + self.yDevsSize)
        
        # We want to keep the raw data intact for response matrix comparison
        if self.axis == 'x':
            newStack = sensorVals*self.scaleFactor
        elif self.axis == 'y':
            newStack = sensorVals*self.scaleFactor
            
        self.signal.emit(newStack)
        

class SpectrogramWidget(pg.PlotWidget):
    read_collected = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, ring, axis):
        super(SpectrogramWidget, self).__init__()
        
        self.ring = ring.lower()
        self.axis = axis.lower()
        self.space = 'BPM'
        
        self.img = pg.ImageItem()
        self.addItem(self.img)

        pos = np.array([0., 1., 1.])
        color = np.array([[0,0,0,255], [0,255,0,255], [255,0,0,255]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos,color)
        pg.colormap
        self.lut = cmap.getLookupTable(0.0, 1.0, 256)

        if self.ring == 'r1':   
            self.deviceX = PT.DeviceProxy('r1/ctl/sofb-01')    
            self.deviceY = PT.DeviceProxy('r1/ctl/sofb-02')
            self.scaleFactor = 0.001 # 2019-08-12 - scale factor updated to same as for r3 after summer BPM update. Was 1000 before. 
            respPath = '/mxn/groups/operators/controlroom/MATLAB/MML/machine/MAXLAB/R1OpsData/ProductionOptics-2019-RP1/GoldenBPMResp_R1_ProductionOptics2019_RP1.mat'
        elif self.ring == 'r3':
            self.deviceX = PT.DeviceProxy('r3/ctl/sofb-01')    
            self.deviceY = PT.DeviceProxy('r3/ctl/sofb-02') 
            self.scaleFactor = 0.001
            respPath = '/mxn/groups/operators/controlroom/MATLAB/MML/machine/MAXLAB/R3OpsData/ProductionOptics-2018-RP2/GoldenBPMResp_R3_ProductionOptics2018_RP2.mat'
            #respPath = '/mxn/groups/operators/controlroom/MATLAB/MML/machine/MAXLAB/R3Data/ProductionOptics-2018-RP2/BPM/BPMvBPMRespMat.mat'

        self.tSize = updateFrequency*minutes*60
        
#        sensorNames = list(self.deviceX.sensor_names)
        self.sensorNamesX = self.deviceX.sensor_names
        self.sensorNamesY = self.deviceY.sensor_names
#        for name in sensorNames:
#            if 'XPos' in name:
#                self.sensorNamesX.append(name)
#            elif 'YPos' in name:
#                self.sensorNamesY.append(name)
        self.xDevsSize = len(self.sensorNamesX)
        self.yDevsSize = len(self.sensorNamesY)

        if self.axis == 'x':
            self.img_array = np.zeros((self.tSize,self.xDevsSize))
            self.ref_array = np.zeros((self.tSize,self.xDevsSize))
            self.Resp = self.loadRespMat(self.deviceX, self.axis)
        elif self.axis == 'y':
            self.img_array = np.zeros((self.tSize,self.yDevsSize))
            self.ref_array = np.zeros((self.tSize,self.yDevsSize))   
            self.Resp = self.loadRespMat(self.deviceY, self.axis)
        self.setLabel('left', self.space + ' - ' + self.axis.upper())
        self.kick_array = np.zeros((self.tSize,self.Resp['R'].shape[1]))
        self.setLabel('bottom', 'time')
        
        self.isoV = pg.InfiniteLine(angle=90, movable=True, pen='y')
        self.addItem(self.isoV, ignoreBounds=False)
        self.isoV.setZValue(1000)
        self.isoV.setPos(self.tSize)
        self.isoV.sigPositionChangeFinished.connect(self.moveLine)

        self.plotGolden()
        self.saveStore()
        
        self.show()
            
    def update(self, newStack):
        """Append new trace to the end of the image"""
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = newStack
        
        self.ref_array = np.roll(self.ref_array, -1, 0)
        self.ref_array[-1:] = self.refSource
        
        self.kick_array = np.roll(self.kick_array, -1, 0)
        self.kick_array[-1:] = self.locateKicks(newStack)

        if 'BPM' in self.space:
            self.img.setImage(np.abs( np.abs(self.img_array) - np.abs(self.ref_array)), autoLevels=False)
        elif 'Corrector' in self.space:
            self.img.setImage(np.abs(self.kick_array), autoLevels=False)

        self.img.setLookupTable(self.lut)
        
#        """Move the vertical line along the image"""
        if round(float(self.isoV.value())) > 0:
            self.isoV.setPos(round(float(self.isoV.value())) - 1.0)
        
#        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
    def plotGolden(self):
        """Plot the image vs. the golden reference orbit"""
        #self.refSource = self.device.sensor_reference_values*self.scaleFactor
        if self.axis == 'x':
            self.refSource = self.deviceX.sensor_reference_values*self.scaleFactor
            #self.refSource = self.refSource[0:self.xDevsSize]
        elif self.axis == 'y':
            self.refSource = self.deviceY.sensor_reference_values*self.scaleFactor
            #self.refSource = self.refSource[self.xDevsSize:]
        self.updateRefImage()
        self.active = 'Golden'
        
    def plotStore(self):
        """Plot the image vs. the saved reference orbit"""
        self.refSource = self.savedRefSource      
        self.updateRefImage()
        self.active = 'Stored'
        
    def saveStore(self):
        """Use the median from the last minute for reference positions"""
        self.savedRefSource = np.median(self.img_array[-updateFrequency*1*60:, :], 0)
        if self.active == 'Stored':
            self.refSource = self.savedRefSource
            self.updateRefImage()

    def updateRefImage(self):
        """Update the reference image with the chosen reference orbit"""
        for i in range(self.tSize):
            if np.sum(self.img_array[i,:]) != 0:
                self.ref_array[i,:] = self.refSource
                
                """Re-calculate the corrector space image as well"""
                self.kick_array[i,:] = self.locateKicks(self.img_array[i,:])
        
    def moveLine(self):
        """Move the vertical line only to integers"""
        val = round(float(self.isoV.value()))
        if val < 0: 
            self.isoV.setPos(0.0)
        elif val > self.tSize:
            self.isoV.setPos(self.tSize)
            
    def loadRespMat(self, device, axis):
#        """Load BPM response matrix and extract off-diagonal elements"""
#        r = scipy.io.loadmat(path)
#        r = r['Rmat']
#        
#        r3filter = {'M11': 1,
#                    'M12': 2,
#                    'U11': 3,
#                    'U21': 4,
#                    'U31': 5,
#                    'U32': 6,
#                    'U41': 7,
#                    'U51': 8,
#                    'M21': 9,
#                    'M22': 10}
#
#        if axis == 'x':
#            R = r[0]['Data'][0]
#            Delta = r[0]['ActuatorDelta'][0][0]
#            DevList = r[0]['Monitor'][0][0][0][3]
#            self.actuatorList = r[0]['Actuator'][0][0][0][3]
#            
#            """Convert the feedback sensors into a corresponding device list"""
#            SensorDevList = np.ndarray(shape=(self.xDevsSize, 2))
#            for i in range(self.xDevsSize):
#                SensorDevList[i] = np.array(re.findall('\d\d+', self.sensorNamesX[i]))
#                if self.ring == 'r3':
#                    SensorDevList[i][0] -= 300
#                    ach = re.findall('[MU]\d', self.sensorNamesX[i])[0]
#                    SensorDevList[i][1] = r3filter[ach + str(int(SensorDevList[i][1]))]
#                elif self.ring == 'r1':
#                    SensorDevList[i][0] -= 100
#
#        elif axis == 'y':
#            R = r[1]['Data'][1]
#            Delta = r[1]['ActuatorDelta'][1][0]
#            DevList = r[1]['Monitor'][1][0][0][3]
#            self.actuatorList = r[1]['Actuator'][1][0][0][3]
#
#            """Convert the feedback sensors into a corresponding device list"""
#            SensorDevList = np.ndarray(shape=(self.yDevsSize, 2))
#            for i in range(self.yDevsSize):
#                SensorDevList[i] = np.array(re.findall('\d\d+', self.sensorNamesY[i]))
#                if self.ring == 'r3':
#                    SensorDevList[i][0] -= 300
#                    ach = re.findall('[MU]\d', self.sensorNamesY[i])[0]
#                    SensorDevList[i][1] = r3filter[ach+str(int(SensorDevList[i][1]))]
#                elif self.ring == 'r1':
#                    SensorDevList[i][0] -= 100
#
#        """Remove BPMs not in use from the response matrix"""
#        FilterList = np.zeros(len(DevList), dtype=bool)
#        for i in range(len(DevList)):
#            for fil in SensorDevList:
#                if (DevList[i,:] == fil).all():
#                    FilterList[i] = True
#                    break
#        R = R[FilterList, :] #/ Delta
    
        R = device.response_matrix
        self.rawActuatorList = device.actuator_names
        self.actuatorList = []
        
        for name in self.rawActuatorList:
            if '/Current' in name:
                self.actuatorList.append(name.replace('/Current',''))
            else:
                self.actuatorList.append(name)
            
            

        """Singular value decomposition"""
        [U, S, Vh] = np.linalg.svd(R, full_matrices=False, compute_uv=True)

        # Sanity check to see if the decomposed matrix is close to the original
        #X_a = np.dot(np.dot(U, np.diag(S)), Vh)
        #print np.std(R), np.std(X_a), np.std(R-X_a)
        #print np.isclose(R, X_a).all()
        
        #//PJB
        #our numpy seems to give u * np.diag(s) * v not vT so
        #R-1 -- typically written V invS UT -- is VT invS UT for us!
        
        """Invert response matrix"""
        # Use the singular values to construct the S matrix
        SVdiag = np.diag(S)

        # Invert it
        SVinv = np.linalg.inv(SVdiag)

        # Extract diagonal from inverted S matrix
        SVinvDiag = np.diag(SVinv)

#        fig = plt.figure()
#        plt.plot(SVinvDiag)
#        plt.show()

        # Replace high singular values with zeros
        if np.max(SVinvDiag) / np.min(SVinvDiag) > 1e4:
            SVinvDiag[SVinvDiag > 1e-4] = 0   # Threshold based on R3 vertical

            #SVmax = len(SVinvDiag)
            #m = 0.1
            #SVinvDiag = SVinvDiag[np.abs(SVinvDiag - np.mean(SVinvDiag)) < m * np.std(SVinvDiag)]
            #SVinvDiag.resize((SVmax,), refcheck=False)

        # Reassemble inverted S matrix
        Sinv = np.diag(SVinvDiag)

        # Used only for plotting it?
        UT = U.T
        invS_UT = Sinv.dot(UT)
        V_invS_UT = (Vh.T).dot(invS_UT)
        weighted_resp_matrix_inv = V_invS_UT

#        fig = plt.figure()
#        plt.imshow(weighted_resp_matrix_inv, origin='lower')
#        plt.show()

        Resp = {'R': R, 
                'U': U,
                'S': S,
                'Vh': Vh,
                'Sinv': Sinv} #'Units': r[0]['Units'][1][0]
        
        return Resp

    def locateKicks(self, stack):
        """Run every new trace through the response matrix"""
        arr = (stack - self.refSource) / self.scaleFactor
        
        #if self.Resp['Units'] == 'Physics':
        #    arr /= 1e9

        # Based on PBs TFB routine
        UT_delta = np.dot(self.Resp['U'].T, arr)
        Sinv_UT_delta = np.dot(self.Resp['Sinv'], UT_delta)
        V_Sinv_UT_delta = np.dot(self.Resp['Vh'].T, Sinv_UT_delta)

        return V_Sinv_UT_delta
        
    def plotTrace(self):
        """Extract the trace at the vertical line and plot it"""
        fig = plt.figure()
        i = round(float(self.isoV.value()))
        if 'BPM' in self.space:
            arr = (self.img_array[i,:] - self.refSource)
        elif 'Corrector' in self.space:
            arr = self.kick_array[i,:]
        plt.plot(arr, picker=5)
        s = (updateFrequency*minutes*60 - i) / updateFrequency
        t = datetime.now() - timedelta(seconds=s)
        plt.title(self.space + ' ' + self.axis.upper() + t.strftime(':   %y-%m-%d %H:%M:%S'))
        plt.xlabel(self.space + ' ' + self.axis.upper() + ' #')
        if 'BPM' in self.space:
            plt.ylabel("Amplitude [um]")
        elif 'Corrector' in self.space:
            plt.ylabel("Kick strength [A]")

        fig.canvas.mpl_connect('pick_event', self.onpick)
        plt.show()
        
    def onpick(self, event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        
        """If more than one data point is selected, compare their values"""
        ind = event.ind
        if len(ind) > 1:
            ind = ind[np.argmax(abs(ydata[ind]))]

        if 'BPM' in self.space:
            if self.axis == 'x':
                s = ' ' + '/'.join(self.sensorNamesX[ind].split('/')[:-1])
            elif self.axis == 'y':
                s = ' ' + '/'.join(self.sensorNamesY[ind].split('/')[:-1])
        elif 'Corrector' in self.space:
            s = ' ' + str(self.actuatorList[int(ind)])

        plt.text(xdata[ind], ydata[ind], s)
        event.canvas.draw()
        
    def toggleSpace(self):
        """Change between showing BPM or corrector space"""
        if 'BPM' in self.space:
            self.space = 'Corrector'
        elif 'Corrector' in self.space:
            self.space = 'BPM'
        self.setLabel('left', self.space + ' - ' + self.axis.upper())
        
    def getData(self):
        return self.img_array, self.kick_array, self.refSource

class BPM2Dtrend(QtGui.QWidget):
    def __init__(self, ring):
        QtGui.QWidget.__init__(self)  
        self.ring = ring
        
        self.wX = SpectrogramWidget(ring, 'x')
        self.wY = SpectrogramWidget(ring, 'y')
        self.wX.read_collected.connect(self.wX.update)
        self.wY.read_collected.connect(self.wY.update)
        
        """Customize the X-tick labels"""
        x = []
        for minute in range(-minutes, minutes/10, 2):
            if minute == 0:
                x.append('Now')
            else:
                x.append(format('%s min') % minute)
        xt = [i for i in range(0, updateFrequency*(minutes+1)*60, updateFrequency*2*60)]
        ticks = [list(zip(xt, x))]
        wXAxis = self.wX.getAxis('bottom')
        wYAxis = self.wY.getAxis('bottom')
        wXAxis.setTicks(ticks)
        wYAxis.setTicks(ticks)
        
        self.dStreamX = DataReader(self.wX.read_collected, ring, 'x') 
        self.dStreamY = DataReader(self.wY.read_collected, ring, 'y') 
        
        self.histX = pg.HistogramLUTWidget()
        self.histX.setImageItem(self.wX.img)
        self.histX.setMinimumWidth(150)
        self.histY = pg.HistogramLUTWidget()
        self.histY.setImageItem(self.wY.img)
        self.histY.setMinimumWidth(150)
        
        self.bpmHistLevelsX = (0.1, 0.3)
        self.bpmHistLevelsY = (0.1, 0.3)
        self.corHistLevelsX = (0.0, 0.0005)
        self.corHistLevelsY = (0.0, 0.0010)

        self.histX.setLevels(self.bpmHistLevelsX[0], self.bpmHistLevelsX[1])
        self.histX.setHistogramRange(0, 0.5)
        self.histY.setLevels(self.bpmHistLevelsY[0], self.bpmHistLevelsY[1])
        self.histY.setHistogramRange(0, 0.5)
        
        updateGolden = QtGui.QPushButton("Plot vs. Golden")
        updateGolden.clicked.connect(self.wX.plotGolden)
        updateGolden.clicked.connect(self.wY.plotGolden)
        updateGolden.setStyleSheet(activeStyle)
        updateGolden.setMinimumWidth(120)
        
        updateStored = QtGui.QPushButton("Plot vs. Stored")
        updateStored.clicked.connect(self.wX.plotStore)
        updateStored.clicked.connect(self.wY.plotStore)
        updateStored.setMinimumWidth(120)

        """Change the background color of the active reference orbit"""
        updateGolden.clicked.connect(lambda: updateGolden.setStyleSheet(activeStyle))
        updateGolden.clicked.connect(lambda: updateStored.setStyleSheet(""))
        updateStored.clicked.connect(lambda: updateGolden.setStyleSheet(""))
        updateStored.clicked.connect(lambda: updateStored.setStyleSheet(activeStyle))

        saveStored = QtGui.QPushButton("Store Current Positions")
        saveStored.clicked.connect(self.wX.saveStore)
        saveStored.clicked.connect(self.wY.saveStore)

        saveLab = QtGui.QLabel("Positions stored:\n" + datetime.now().strftime('%y-%m-%d %H:%M:%S'))
        saveStored.clicked.connect(lambda: saveLab.setText("Positions stored:\n" + datetime.now().strftime('%y-%m-%d %H:%M:%S')))
        
        self.space = QtGui.QPushButton("BPM space")
        self.space.clicked.connect(self.wX.toggleSpace)
        self.space.clicked.connect(self.wY.toggleSpace)
        self.space.clicked.connect(self.toggleSpace)
        self.space.setMinimumWidth(120)
        self.space.setStyleSheet(bpmStyle)
        
        plotTraceX = QtGui.QPushButton("Plot trace X")
        plotTraceX.clicked.connect(self.wX.plotTrace)
        plotTraceY = QtGui.QPushButton("Plot trace Y")
        plotTraceY.clicked.connect(self.wY.plotTrace)
        
        saveMatButton = QtGui.QPushButton("Save .mat data")
        saveMatButton.clicked.connect(self.saveMat)
        
        self.pauseButton = QtGui.QPushButton("Pause")
        self.pauseButton.clicked.connect(self.pauseResume)
        self.pauseButton.setMinimumWidth(100)
        
        self.clockLab = QtGui.QLabel(datetime.now().strftime('%y-%m-%d %H:%M:%S'))
        
        hBox1 = QtGui.QHBoxLayout()
        hBox1.addWidget(self.wX)
        hBox1.addWidget(self.histX)
        
        hBox2 = QtGui.QHBoxLayout()
        hBox2.addWidget(self.wY)
        hBox2.addWidget(self.histY)
        
        hBox3 = QtGui.QHBoxLayout()
        hBox3.addWidget(updateGolden)
        hBox3.addWidget(updateStored)
        hBox3.addWidget(saveLab)
        hBox3.addWidget(saveStored)
        hBox3.addItem(QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        hBox3.addWidget(self.space)
        hBox3.addWidget(plotTraceX)
        hBox3.addWidget(plotTraceY)
        hBox3.addItem(QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        hBox3.addWidget(saveMatButton)
        hBox3.addWidget(self.pauseButton)
        hBox3.addWidget(self.clockLab)

        vBox = QtGui.QVBoxLayout()
        vBox.addLayout(hBox1)
        vBox.addLayout(hBox2)
        vBox.addLayout(hBox3)
        
        self.setLayout(vBox)
        self.setWindowTitle(ring.upper() + " BPM trends")
        self.setMinimumHeight(600)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.dStreamX.read)
        self.timer.timeout.connect(self.dStreamY.read)
        self.timer.timeout.connect(self.clock)
        self.timer.start(1000/updateFrequency)
        
        """Automatically update the reference orbit after a minute"""
        oneShot = QtCore.QTimer()
        oneShot.singleShot(1000*60, saveStored.click)
        
    def toggleSpace(self):
        if 'BPM' in str(self.space.text()):
            self.space.setText('Corrector space')
            self.space.setStyleSheet(corStyle)
            """Store current histogram levels for this space"""
            self.bpmHistLevelsX = self.histX.getLevels()
            self.bpmHistLevelsY = self.histY.getLevels()
            """Change histogram levels for new space"""
            self.histX.setLevels(self.corHistLevelsX[0], self.corHistLevelsX[1])
            self.histY.setLevels(self.corHistLevelsY[0], self.corHistLevelsY[1])
            self.histX.setHistogramRange(0, 0.001)
            self.histY.setHistogramRange(0, 0.002)
            
        elif 'Corrector' in str(self.space.text()):
            self.space.setText('BPM space')
            self.space.setStyleSheet(bpmStyle)
            """Store current histogram levels for this space"""
            self.corHistLevelsX = self.histX.getLevels()
            self.corHistLevelsY = self.histY.getLevels()
            """Change histogram levels for new space"""
            self.histX.setLevels(self.bpmHistLevelsX[0], self.bpmHistLevelsX[1])
            self.histY.setLevels(self.bpmHistLevelsY[0], self.bpmHistLevelsY[1])
            self.histX.setHistogramRange(0, 0.5)
            self.histY.setHistogramRange(0, 0.5)
            
    def saveMat(self):
        """Pause data acquisition while saving"""
        stopFlag = False
        if self.timer.isActive():
            self.timer.stop()
            stopFlag = True

        Xbpm, Xcorr, Xref = self.wX.getData()
        Ybpm, Ycorr, Yref = self.wY.getData()

        tmpDict = dict(Xbpm=Xbpm,
                       Xcorr=Xcorr,
                       Xref=Xref,
                       Ybpm=Ybpm,
                       Ycorr=Ycorr,
                       Yref=Yref)
                       
        direc = '/mxn/groups/operators/controlroom/data/2D_BPM_data_dumps/'
        filename = self.ring.upper() + '_2D_BPM_dump_' + datetime.now().strftime('%y-%m-%d_%H-%M-%S')
        
        scipy.io.savemat(direc + filename, tmpDict, appendmat=True)
        
        """Open folder containing the data dump"""
        os.system('xdg-open "%s"' % direc)
            
        if stopFlag:
            self.timer.start(1000/updateFrequency)
        
    def pauseResume(self):
        if self.timer.isActive():
            self.pauseButton.setStyleSheet(stopStyle)
            self.timer.stop()
        else:
            self.pauseButton.setStyleSheet("")
            self.timer.start(1000/updateFrequency)
            
    def clock(self):
        self.clockLab.setText(datetime.now().strftime('%y-%m-%d %H:%M:%S'))
            
    def closeEvent(self, event):
        """Close any open kick plots"""
        plt.close("all")
        

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)    
    
    win = BPM2Dtrend(sys.argv[1])
    win.show() 

    sys.exit(app.exec_())

    
    
