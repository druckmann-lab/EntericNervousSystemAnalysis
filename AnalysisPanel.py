import spatial_analysis_utils as utils0
import numpy as np
import pandas as pd
import copy, os
import matplotlib.pyplot as plt
from PIL import Image 
import utils
from matplotlib import gridspec
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
class AnalysisPanel:
    def __init__(self,dataPath='../Data/',nBinsInterpolation=1000, parts=None, windowSize=[0.25, 0.25]):
        # windowSize == [windowSizeHorizontal, windowSizeVertical]
        # if parts is None, then the program will automatically fetch the names of the parts from the file names in dataPath.
        self.info = {}
        self.dataPath = dataPath
        if parts is not None:
            self.parts = parts
        else:
            self.fetchPartsFromDataPath()
        
        self.nBinsInterpolation = nBinsInterpolation
        for part in self.parts:
            self.info[part] = {}
            dataFile = os.path.join(self.dataPath,"Collated {} coords.csv".format(part))
            dataFileFieldSize = os.path.join(self.dataPath,"Collated {} field size.csv".format(part))
            self.info[part]['dataDf'] = pd.read_csv(dataFile)
            self.info[part]['dataFieldSizeDf'] = pd.read_csv(dataFileFieldSize)
        self.windowSize = windowSize
        self.setDefaultYCutRatioForWidth()
        self.setDefaultSmoothSigma()
        self.setDefaultMinimalDistanceForSpatialRand()
        self.prep()
        self.cal_scale_info(False)
        self.global_norm()



    def fetchPartsFromDataPath(self):
        parts = []
        target = 'coords.csv'
        for fileName in os.listdir(self.dataPath):
            if fileName[-len(target):] == target:
                parts.append(fileName.split(' ')[1])
        self.parts = sorted(list(set(parts)))
        assert len(self.parts), 'No relavant csv files. Please check dataPath.'

    def cal_scale_info(self, verbose=True):
        mx_arr = []
        mi_arr = []
        for part in self.parts:
            for tissueIndex, tissue in enumerate(self.info[part]['tissue']):
                arr = self.info[part]['dataArray'][tissueIndex]
                if verbose:
                    print('{} {} {:.0f} {:.0f}'.format(
                        part, tissue, arr[:,0].max()-arr[:,0].min(), arr[:,1].max()-arr[:,1].min()
                        ))
                mx_arr.append([arr[:,0].max(), arr[:,1].max()])
                mi_arr.append([arr[:,0].min(), arr[:,1].min()])
        mx_arr = np.array(mx_arr) #(n_tissue, 2)
        mi_arr = np.array(mi_arr)
        self.scale_factor = (mx_arr - mi_arr).max(axis=0)
        # self.scale_factor_b = mi_arr.min(axis=0)
        '''
        Typical examples
        small side 1584 1670
        big side 2499 2500
        So we can set the grid size to be 25
        '''
    def global_norm(self):
        for part in self.parts:
            self.info[part]['normGlobalInfo'] = []
            for tissueIndex, tissue in enumerate(self.info[part]['tissue']):

                arr = copy.deepcopy(self.info[part]['dataArray'][tissueIndex])
                arr = arr / self.scale_factor
                # center the data to 0.5
                arr = arr - arr.mean(axis=0)
                arr = arr + 0.5

                self.info[part]['normGlobalArray'].append(arr)


    def prep(self):
        unit = {
                'regionType':[],
                'dataArray':[],
                'dataFieldSize':[],
                'dataFieldSizeInMicron':[],
                'normLocArray':[],
                'normGlobalArray':[],
                'tissue':[],
                }
        for part in self.parts:
            self.info[part].update(copy.deepcopy(unit))
            data, dataFieldSizeDf = self.info[part]['dataDf'], self.info[part]['dataFieldSizeDf']
            for tissue in data.Tissue.unique():
                self.info[part]['regionType'].append(tissue)
                self.info[part]['dataArray'].append(data.loc[data.Tissue==tissue,['X','Y']].to_numpy())

                if 'Field X (pixels)' in dataFieldSizeDf.columns:
                    targetColumn = [['Field X (pixels)','Field Y (pixels)'],['Field X (um)','Field Y (um)']]
                else:
                    # Lori's data has different column names. Here is a work-around.
                    targetColumn = [['X (pixel)','Y (pixel)'],['X (micron)','Y (micron)']]
                

                self.info[part]['dataFieldSize'].append(
                    dataFieldSizeDf.loc[dataFieldSizeDf.Tissue==tissue,targetColumn[0]].to_numpy()
                    )

                self.info[part]['dataFieldSizeInMicron'].append(
                    dataFieldSizeDf.loc[dataFieldSizeDf.Tissue==tissue,targetColumn[1]].to_numpy()
                    )


                arr = copy.deepcopy(self.info[part]['dataArray'][-1])
                self.info[part]['normLocArray'].append(utils0.simple_normalize_data(arr))

                self.info[part]['tissue'].append(tissue)


    def latticeDataArray(self,nGridGlobal=300,nGridLoc=300,yGridScale=1):
        # This method is not being actively used right now
        self.nGridGlobal = nGridGlobal
        self.nGridLoc = nGridLoc
        self.xGridScale= xGridScale
        nGrid = {
            'Global':nGridGlobal,
            'Loc':nGridLoc
            }

        for feat in ['Global','Loc']:
            xEdges = np.linspace(0,1,nGrid[feat])
            yEdges = np.linspace(0,1,nGrid[feat]*yGridScale)
            for part in self.parts:
                latticeK = 'norm{}Lattice'.format(feat)
                arrK = 'norm{}Array'.format(feat)
                self.info[part][latticeK] = []
                for tissueIndex, tissue in enumerate(self.info[part]['tissue']):
                    lattice = np.zeros((50,50))
                    arr = self.info[part][arrK][tissueIndex]

                    hist, _, _ = np.histogram2d(arr[:,0], arr[:,1], bins=(xEdges, yEdges))
                    self.info[part][latticeK].append(hist)
                self.info[part][latticeK] = np.array(self.info[part][latticeK])

                arr = self.info[part][latticeK]

                # arr[:,:-1] += arr[:,1]

    def plotNeuronLattice(self, part='PC', tissueIndex=14, feat='Global'):
        latticeK='norm{}Lattice'.format(feat)
        plt.figure(figsize=(10,10))
        print(self.info[part]['normGlobalArray'][tissueIndex].shape)
        x = (self.info[part][latticeK][tissueIndex]>0)*254
        # x = ndimage.gaussian_filter(x, 3)
        # x[x<50] = 0
        plt.imshow(x.T, origin='lower')


    def dataSpatialRandomnessComparisonSummary(self,  part='duodenum', tissueIndex=0,binPointNum=50, sampleNumForRandomComparison=500, doPlot = True, fontsize = 15, doReturn=False):
        arr = copy.deepcopy(self.info[part]['dataArray'][tissueIndex])
        # arr = arr - arr.min(axis=0)
        # arr = arr - arr.mean(axis=0)
        axs, fig, zData, zRandom, aPercent, zPercent, zRandomMore = utils.data_spatial_randomness_comparison_summary(arr,binPointNum, sampleNumForRandomComparison, minimal_distance=self.defaultMinimalDistance, do_plot = doPlot, fontsize = fontsize)
        if doPlot:
            axs[0].set_title('{}-tissueIndex[{}] Data (Normed)'.format(part, tissueIndex), fontsize = fontsize)
            fig.tight_layout()
        if doReturn:
            return fig, zData, zRandom, aPercent, zPercent, zRandomMore

    def plotNeuronMap(self, part='duodenum', tissueIndex=0, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111)
        arr = copy.deepcopy(self.info[part]['dataArray'][tissueIndex])
        # arr *= self.info[part]['dataFieldSizeInMicron'][tissueIndex][0][0]
        # arr -= arr.mean(axis=0) + 1500
        arr = arr - arr.min(axis=0)
        arr = arr - arr.mean(axis=0) 
        # arr += self.scale_factor
        ax.scatter(arr[:,0], arr[:,1],color='b',s=5)

        ax.set_xlim(-self.scale_factor[0]/2,self.scale_factor[0]/2)
        ax.set_ylim(-self.scale_factor[1]/2,self.scale_factor[1]/2)
        # print(self.info[part]['dataFieldSizeInMicron'][tissueIndex][0][0])
        # ax.axis("off")
        # path = "temp/neuron_map.png"
        # fig.savefig(path)
        # im = np.asarray(Image.open(path)).mean(axis=-1)
        # return im

    def getAllCIH(self, unit_n_step = 100):
        # get conditional intensity histogram
        # This may take some time, because the method is dealing with all the histograms.
        x_size, y_size = self.windowSize
        print('start')
        for part in self.parts:
            print('working on {}'.format(part),end='|')
            for feat in ['Global']:
                arrK = 'norm{}Array'.format(feat)
                self.info[part]['hist{}'.format(feat)] = []
                for tissueIndex, tissue in enumerate(self.info[part]['tissue']):
                    arr = self.info[part][arrK][tissueIndex]
                    hist, xEdges, yEdges = \
                        utils.get_conditional_intensity_histogram_rectangular(arr, x_size, y_size, unit_n_step)
                    self.info[part]['hist{}'.format(feat)].append(hist)
        self.edgeVecs = xEdges, yEdges
        print()
        print('end')

    def displayAllWidthAndDistance(self,yCutRatioForWidthToDisplay=None):
        if yCutRatioForWidthToDisplay is None:
            yCutRatioForWidthToDisplay = [self.defaultYCutRatioForWidth,1.0]
        yCutRatioForDistanceToDisplay = [1.0]
        yCutRatios = sorted(list(set(
            list(yCutRatioForWidthToDisplay) + list(yCutRatioForDistanceToDisplay)
            )))
        self.getAllWidthAndDistance(yCutRatios)
        print('Width')
        for yCutRatio in yCutRatioForWidthToDisplay:
            print('\n\n\nyCutRatio {}'.format(yCutRatio))
            for part in self.parts:
                arr = np.array(self.info[part]['result'][yCutRatio]['width'])
                print(part)
                print('mean {:.2f}   std {:.2f}   median {:.2f}   min {:.2f}   max {:.2f}'.format(
                    arr.mean(),arr.std(),np.median(arr),arr.min(),arr.max()
                    ))
                # print(arr[14])
                print()

        print('\n\n\n\n\n\nDistance')
        for yCutRatio in yCutRatioForDistanceToDisplay:
            print('\n\n\nyCutRatio {}'.format(yCutRatio))
            for part in self.parts:
                arr = np.array(self.info[part]['result'][yCutRatio]['distance'])
                sample = (arr!=None)
                print(part)
                if any(sample):
                    arr = arr[sample]
                    print('mean {:.2f}   std {:.2f}   median {:.2f}   min {:.2f}   max {:.2f}'.format(
                        arr.mean(),arr.std(),np.median(arr),arr.min(),arr.max()
                    ))
                else:
                    print('not available')
                print()
    def getAllWidthAndDistance(self, yCutRatios, feat='Global'):
        for part in self.parts:
            self.info[part]['result'] = {}
            for yCutRatio in yCutRatios:
                self.info[part]['result'][yCutRatio] = {'width':[],'distance':[]}
                for tissueIndex, tissue in enumerate(self.info[part]['tissue']):
                    # if yCutRatio >= 0.5:
                    #     print(part, tissueIndex, tissue)
                    histRaw = copy.deepcopy(self.info[part]['hist{}'.format(feat)][tissueIndex])
                    histRaw = utils.normH(histRaw)
                    midsRaw = [utils.edgeVec2midPoint(self.edgeVecs[i]) for i in range(2)]

                    hist, mids, edgeVecsNow = self.yCutHist1d(yCutRatio, histRaw, midsRaw, ax=None)

                    binCenterVecMicronXY = list(self.process_edgeVecXY(edgeVecsNow, part, tissueIndex,feat))


                    width, distance = utils.doHist1d(binCenterVecMicronXY,hist,mids,ax=None,doSecPeak=yCutRatio>0.4,interpolation_bin_number=self.nBinsInterpolation, smoothSigma=self.defaultSmoothSigma)


                    self.info[part]['result'][yCutRatio]['width'].append(width)
                    self.info[part]['result'][yCutRatio]['distance'].append(distance)

    def plotAllCIH(self,fontsize=15,figFormat='pdf'):
        smooth = self.defaultSmoothSigma
        feat = 'Global'
        print('start')
        for part in self.parts:
            print('working on {}'.format(part),end='|')
            for tissueIndex, tissue in enumerate(self.info[part]['tissue']):
                if smooth > 0:
                    folder = 'log/histDemo/{}Norm-smooth{}-{}'.format(feat, str(round(smooth,2)),part)
                else:
                    folder = 'log/histDemo/{}Norm-{}'.format(feat,part)
                filename = '{}Norm-{}-{}.{}'.format(feat, part, tissueIndex,figFormat)
                fig = self.plotCIH(part=part, tissueIndex=tissueIndex, reutrnFig=True, fontsize=fontsize)
                utils.quick_save_fig(folder, filename,fig)
        print()
        print('end')

    def setDefaultYCutRatioForWidth(self, defaultYCutRatioForWidth=0.1):
        # We can tune the default yCutRatio for width here. This will have effect on the result of plotCIH. Tuning this after getAllCIH saves time.
        self.defaultYCutRatioForWidth = defaultYCutRatioForWidth

    def SetDefaultYCutRatioForWidth(self, defaultYCutRatioForWidth=0.1):
        # For compatibility of a previous version.
        return self.setDefaultYCutRatioForWidth(defaultYCutRatioForWidth)

    def setDefaultMinimalDistanceForSpatialRand(self, defaultMinimalDistance=10):
        self.defaultMinimalDistance = defaultMinimalDistance


    def setDefaultSmoothSigma(self, defaultSmoothSigma=0):
        self.defaultSmoothSigma = defaultSmoothSigma

    def calAllDataSpatialRandomnessComparison(self, binPointNum=50, sampleNumForRandomComparison=500, doPlot=False, figFormat='pdf'):

        if doPlot:
            folder = 'log/spatialRandomnessComparison/binPointNum[{}]-sampleNum[{}]'.format(binPointNum,sampleNumForRandomComparison)
        self.spatialRandomnessComparisonResult = {}
        print('start')
        for part in self.parts:
            print('working on {}'.format(part),end='|')
            self.spatialRandomnessComparisonResult[part] = {}
            for tissueIndex, tissue in enumerate(self.info[part]['tissue']):

                fig, zData, zRandom, aPercent, zPercent, zRandomMore = self.dataSpatialRandomnessComparisonSummary(part, tissueIndex, doReturn=True, doPlot=doPlot, binPointNum=binPointNum, sampleNumForRandomComparison=sampleNumForRandomComparison)
                rst_cell = {
                    'aPercent':aPercent,
                    'zData':zData,
                    'zRandom':zRandom,
                    'zPercent':zPercent,
                    'zRandomMore':zRandomMore
                }
                zDiffMore = rst_cell['zData'] - rst_cell['zRandomMore']
                rst_cell['zDiff'] = np.mean(zDiffMore)
                rst_cell['zDiffStd'] = np.std(zDiffMore)


                self.spatialRandomnessComparisonResult[part][tissueIndex] = rst_cell
                if doPlot:
                    filename = '{}-{}.{}'.format(part, tissueIndex, figFormat)
                    utils.quick_save_fig(folder, filename,fig)
        print()
        print('end')
    def plotAllDataSpatialRandomnessComparison(self, binPointNum=50, sampleNumForRandomComparison=500,figFormat='pdf'):
        self.calAllDataSpatialRandomnessComparison(doPlot=True, binPointNum=binPointNum, sampleNumForRandomComparison=sampleNumForRandomComparison,figFormat=figFormat)

    def plotCIH(self, part='duodenum', tissueIndex=0, yCutRatios=None, doTitle=True, fontsize=15, norm=True, feat='Global', reutrnFig=False, doSecPeakList=[False,True], doMinList=[True,False],smoothSigma=None):

        if smoothSigma is None:
            smoothSigma = self.defaultSmoothSigma

        if yCutRatios is None:
            yCutRatios = [self.defaultYCutRatioForWidth,1.0] 
        height_ratios=[2.5, 2.5*self.windowSize[1]/self.windowSize[0], 0.1]+[1 for _ in yCutRatios]
        fig = plt.figure(figsize = (6,2.5*np.sum(height_ratios)))
        gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios)
        axs = [plt.subplot(gs[i]) for i in range(3+len(yCutRatios))]
        self.plotNeuronMap(part, tissueIndex, axs[0])

        for ax in axs:
            ax.tick_params(labelsize=fontsize)
        if doTitle:
            axs[1].set_title('{}   {} (index {})'.format(part,self.info[part]['tissue'][tissueIndex],tissueIndex), fontsize=fontsize)
        axs[2].axvline(0,color='k',linestyle='--')

        histRaw = copy.deepcopy(self.info[part]['hist{}'.format(feat)][tissueIndex])

        if norm:
            histRaw = utils.normH(histRaw)
        im = utils.plot_conditional_intensity_histogram(
            histRaw.T,
            list(self.process_edgeVecXY(self.edgeVecs, part, tissueIndex,feat)),
            axs[1],
            fontsize=fontsize
            )

        fig.colorbar(im, cax=axs[2],orientation='horizontal')


        midsRaw = [utils.edgeVec2midPoint(self.edgeVecs[i]) for i in range(2)]

        for ii, yCutRatio in enumerate(yCutRatios):
            hist, mids, edgeVecsNow = self.yCutHist1d(yCutRatio, histRaw, midsRaw, ax=axs[1])
            binCenterVecMicronXY = list(self.process_edgeVecXY(edgeVecsNow, part, tissueIndex,feat))
            utils.doHist1d(binCenterVecMicronXY,hist,mids,ax=axs[3+ii],doTitle=doTitle, fontsize=fontsize, titleAdd='yCutRatio{}; '.format(yCutRatio),doSecPeak=doSecPeakList[ii], doMin=doMinList[ii], interpolation_bin_number=self.nBinsInterpolation, smoothSigma=smoothSigma)

        fig.tight_layout()

        if reutrnFig:
            return fig

    def yCutHist1d(self, yCutRatio, histRaw, midsRaw, ax=None):
        if yCutRatio < 1:
            offset = int(yCutRatio * len(self.edgeVecs[1])/2 + 0.5)
            edgeVecsNow = self.edgeVecs[0], self.edgeVecs[1][midsRaw[1]-offset:midsRaw[1]+offset]
            hist = histRaw[:,midsRaw[1]-offset+1:midsRaw[1]+offset]
            mids = [utils.edgeVec2midPoint(edgeVecsNow[i]) for i in range(2)]
            if ax is not None:
                ax.axhline(midsRaw[1]-offset,color='w')
                ax.axhline(midsRaw[1]+offset,color='w')
        else:
            hist = histRaw
            mids = midsRaw
            edgeVecsNow = self.edgeVecs
        return hist, mids, edgeVecsNow

    def displaySpatialRandomnessComparisonResult(self):
        rst = self.spatialRandomnessComparisonResult
        print('Part   aPercent   zData    zRandMean    zDiff    zDiffStd    zDiffSem')
        for part in rst.keys():
            
            for tissueIndex in rst[part].keys():
                rst_ = rst[part][tissueIndex]
                # rst_['aPercent'] = 1 - (rst_['aPercent']/100 - 1)
                print("{}      {:.2f}   {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}".format(part,
                    100*rst_['aPercent'],
                    rst_['zData'],
                    np.mean(rst_['zRandomMore']),
                    rst_['zDiff'],
                    rst_['zDiffStd'],
                    rst_['zDiffStd']/np.sqrt(len(rst_['zRandomMore']))))
                print()


    def compare(self, targetInfos=[['duodenum',14, 0.3],['PC',14, 0.3]]):
        f = plt.figure(figsize = (6,4.5))
        for targetInfo in targetInfos:
            part, tissueIndex, yCutRatio = targetInfo
            hist = copy.deepcopy(self.info[part]['histLoc'][tissueIndex])
            mids = [utils.edgeVec2midPoint(self.edgeVecs[i]) for i in range(2)]

            edgeVecsNow = self.edgeVecs
            if yCutRatio < 1:
                offset = int(yCutRatio * len(self.edgeVecs[1])/2 + 0.5)
                edgeVecsNow = self.edgeVecs[0], self.edgeVecs[1][mids[1]-offset:mids[1]+offset]
                hist = hist[:,mids[1]-offset+1:mids[1]+offset]
                mids = [utils.edgeVec2midPoint(edgeVecsNow[i]) for i in range(2)]
            hist1d = utils.histToHist1d(hist, mids)

            plt.plot(utils.edgeVec2binCenterVec(edgeVecsNow[0]),hist1d)
    def process_edgeVecXY(self, edgeVecs, part='duodenum', tissueIndex=0, feat='Loc'):

        for i in [0,1]:
            yield self.process_edgeVec(edgeVecs[i], part, tissueIndex, feat, i)

    def process_edgeVec(self, edgeVec, part, tissueIndex, feat, x_or_y):

        binCenterVec = utils.edgeVec2binCenterVec(edgeVec)
        if feat == 'Loc':
            binCenterVecMicron = binCenterVec*(self.info[part]['dataFieldSizeInMicron'][tissueIndex][0][0])
        else:
            binCenterVecMicron = binCenterVec*self.scale_factor[x_or_y]
        return binCenterVecMicron