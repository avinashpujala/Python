# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Importing somer basics
import numpy as np
import matplotlib.pyplot as plt
import os, time, scipy, sys

codeDir = [r'C:\Users\pujalaa\Documents\Code\Python\code\codeFromNV', 
           r'C:\Users\pujalaa\Documents\Code\Python\code\util']
for item in codeDir:
    sys.path.insert(1,item)


#%% Data directory
inputDir = r'S:\Avinash\Ablations and behavior\GrpData\Session 20170303_Clustering and Regression of Alx'
fName = 'Cell data for clustering_20170303T070008_.mat'

#%% Pointing to cell data
import h5py
file = h5py.File(os.path.join(inputDir,fName), mode = 'r', libver='latest')
data_disk = file[list(file.keys())[1]]
print('Imported "' + list(file.keys())[1]+ '" with they keys...')
print(list(data_disk.keys()))

#%% Importing data into workspace
import SignalProcessingTools as spt
tic = time.time()
print('Reading...')
sys.stdout.flush()
data = {}
for key in data_disk.keys():    
    print(key),sys.stdout.flush()       
    if 'dataset' in str(data_disk[key]).lower():        
        data[key] = data_disk[key][:]
    else:
        data[key] = {}
        for key_key in data_disk[key].keys():           
            print('\t', key_key), sys.stdout.flush()            
            data[key][key_key] = data_disk[key][key_key][:]
blah = {}
print('Reading object references...')
for key in data['info']:
    item = data['info'][key]
    blah[key]=[]
    print(key)
    #sys.stdout.flush()
    for val in item:
        if isinstance(val[0],h5py.Reference):
            blah[key].append(data_disk[val[0]][:])
    try:
        blah[key] = np.array(blah[key]).ravel()
    except:
        pass
data['info'] = blah

data['inds']['inMask'] = data['inds']['inMask'].astype(int)[0]

print('\n', int(time.time()-tic),'sec')
print(time.ctime())

#data['avg'] = spt.standardize(data['avg'])

#%% Kmeans
from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

n_clusters = 16
random_state = 153

tic = time.time()
data['kmeans'] = {}
data['kmeans']['X'] = np.squeeze(data['dFF'][:,data['inds']['inMask'].astype(int)].transpose())
data['kmeans']['kmeans'] = KMeans(n_clusters,init= 'k-means++',random_state = random_state, n_init = 10)
print(data['kmeans']['kmeans'])
out = data['kmeans']['kmeans'].fit(data['kmeans']['X'])
centroids = out.cluster_centers_
labels = out.labels_
data['kmeans']['distFromCentroids'] = out.transform(data['kmeans']['X'])

print(int(time.time()-tic) , 'sec')
print(time.ctime())

#%% Plotting K-means
#cMap = plt.cm.get_cmap('jet')
#sb.set_palette('Paired',n_clusters)
#sb.set_style('white')
plt.style.use('dark_background')
plt.figure(figsize = [14,10])
yt = []
color_idx = np.linspace(0,1,n_clusters)
for cNum, centroid in enumerate(centroids):
    yt.append(cNum*-15)
    plt.plot(data['time'],scale(centroid)+yt[cNum], color = plt.cm.RdYlGn(color_idx[cNum]))
plt.xlim(data['time'].min(), data['time'].max())
plt.yticks(yt, np.arange(n_clusters)+1)
plt.xlabel('Time (sec)',fontsize = 16)
plt.ylabel('Cluster #', fontsize = 16)
plt.box('off')
plt.title('Cluster centroids', fontsize = 20)
plt.tick_params(labelsize = 14);

#%% Plotting some more stuff
plt.figure(figsize = (14,14))
#plt.tight_layout
#plt.subplot(211)
plt.imshow(data['kmeans']['X'],cmap = plt.cm.jet)
plt.axis('auto')
plt.clim(-0.1,0.5)
xtl = np.arange(0,data['time'][-1],500).astype(int)
#xtl = (np.floor(np.linspace(data['time'][0],data['time'][-1],10)/100)*100).astype(int) 
xt = xtl/(data['time'][1]-data['time'][0])
ax  = plt.gca()
ax.xaxis.set_ticks(xt)
ax.xaxis.set_ticklabels(xtl)
ax.tick_params(labelsize = 14)
plt.ylabel('Cell #', fontsize = 16)
plt.xlabel('Time (sec)', fontsize = 16)
plt.title('Cell data, Unsorted', fontsize = 18);

plt.figure(figsize = (14,14))
#plt.subplot(212)
labels_sort =  np.sort(labels)
labels_sort_inds = np.argsort(labels)
data_clstr_sort = data['kmeans']['X'][labels_sort_inds,:]
plt.imshow(data_clstr_sort,cmap = 'jet')
plt.axis('auto')
plt.clim(-0.1, 0.5)

lbl_ctrs = []
for lbl in np.unique(labels):
    inds = np.where(labels_sort==lbl)[0]
    lbl_ctrs.append(np.median(inds).astype(int))
ax = plt.gca()
ax.yaxis.set_ticks(lbl_ctrs)
ax.yaxis.set_ticklabels(np.unique(labels)+1)

xtl = np.arange(0,data['time'][-1],500).astype(int)
#xtl = (np.floor(np.linspace(data['time'][0],data['time'][-1],10)/100)*100).astype(int) 
xt = xtl/(data['time'][1]-data['time'][0])
ax.xaxis.set_ticks(xt)
ax.xaxis.set_ticklabels(xtl)

ax.tick_params(labelsize = 14)
plt.ylabel('Clstr #', fontsize = 16)
plt.xlabel('Time (sec)', fontsize = 16)
plt.title('Cell data, sorted by cluster', fontsize = 18);

#%% An adjustment for some datasets
data['avg'] = np.transpose(data['avg'],[0,2,1])
np.shape(data['avg'])

#%% Create rgb img stack with cells colored by cluster ID
import tifffile as tff
import volTools as volt

imgStack_norm = spt.standardize(data['avg'].copy())
imgStack = np.transpose(np.tile(np.zeros(np.shape(imgStack_norm)),[3,1,1,1]),[1,2,3,0])
sliceList = np.arange(np.shape(data['avg'])[0])
#sliceList = [30]
for z in sliceList:   
    inds_slice = np.where(data['info']['slice'].astype(int)==z+1)[0]    
    imgStack[z,:,:,:] = volt.gray2rgb(imgStack_norm[z])
    for lbl in np.unique(labels):        
        inds_lbl = data['inds']['inMask'].ravel()[np.where(labels == lbl)[0]].astype(int)
        inds_cell = np.intersect1d(inds_slice,inds_lbl)-1      
        if len(inds_cell)>0:
            pxls_cell = [data['info']['inds'][ind] for ind in inds_cell]
            #pxls = np.squeeze(np.array(pxls_cell)).ravel().astype(int)
            rgb= plt.cm.RdYlGn(color_idx[lbl])[0:3]           
            for pxl in pxls_cell:                
                pxl = pxl.ravel().astype(int)
                pxl = np.unravel_index(pxl,np.shape(data['avg'][0].transpose()))
                pxl_zip = list(zip(pxl[0],pxl[1]))                
                for valNum,val in enumerate(rgb):                   
                    imgStack[z,pxl[1],pxl[0],valNum] = val
imgStack= np.transpose(imgStack,[0,2,1,3])

#%% Display a slice to check
z = 20
plt.figure(figsize = (16,16))
plt.imshow(imgStack[z])
plt.axis('off')
plt.title('Clustered cells, slice # ' + str(z));

print(time.ctime())


#%% Save img stack in specified dir
# saveDir = r'S:\Avinash\Ablations and behavior\GrpData\Session 20170121\blah'
saveDir = inputDir
imgName = 'K-means clustering of Alx cells_clstrs unord' + '.tif'
tff.imsave(os.path.join(saveDir,imgName),np.transpose(imgStack,[0,3,1,2]))


#%% REGRESSION
import AnalyzeEphysData as aed

# Chose # 2 after visual inspection (Note: Run Kmeans with k-means++ 
#  init and unordered clusters before runing this)
centroid_M = centroids[4,:]

dt = data['time'][2] - data['time'][1]
centroid_M = spt.zscore(spt.chebFilt(centroid_M,dt,0.01,btype='high'))
pks = spt.findPeaks(centroid_M,thr= 3, minPkDist=30)

plt.figure(figsize = (14,6))
plt.style.use('dark_background')
plt.subplot(131)
plt.plot(data['time'],centroid_M)
plt.plot(data['time'][pks[0]],centroid_M[pks[0]],'ro')
plt.xlim(100,500)
plt.xlabel('Time (s)')
plt.ylabel('dF/F')
plt.title('Some stim-elicited responses in chosen centroid' 
          '\n Shown peaks used to average responses')
centroid_M_seg = aed.SegmentDataByEvents(centroid_M,pks[0],20,50,axis =0)
trlLens = np.array([len(trl) for trl in centroid_M_seg])
shortTrls = np.where(trlLens < scipy.stats.mode(trlLens)[0])[0]
centroid_M_seg = np.delete(centroid_M_seg,shortTrls,axis = 0)
centroid_M_avg = spt.zscore(np.mean(centroid_M_seg,0))

#plt.figure()
plt.subplot(132)
plt.plot(centroid_M_avg)
plt.title('Average response');

pk = spt.findPeaks(centroid_M_avg,thr = 1.5)
pk = pk[0][np.where(pk[1]==np.max(pk[1]))[0]]
pk = pk[0]
on  = spt.findPeaks(spt.zscore(np.diff(centroid_M_avg)),thr =2)
on = on[0][on[0]<pk]
on = on[-1]-3
plt.plot(on,centroid_M_avg[on],'go')
centroid_M_avg = centroid_M_avg[on:-1]

cirf_raw = spt.standardize(centroid_M_avg)
cirf_raw = cirf_raw-cirf_raw[0]

#plt.figure()
plt.subplot(133)
cirf_time = np.arange(0,len(cirf_raw))*dt
plt.plot(cirf_time,cirf_raw)
plt.title('Raw CIRF');

#%% First pass manual fit, then pass as initial parameters to curv_fit
p = [2.5,4.8,1,1] # [tau_rise, tau_decay, amplitude, translation]
epsp = spt.generateEPSP(cirf_time,p[0],p[1],p[2],p[3])
plt.style.use('seaborn-deep')
plt.figure(figsize = (14,8))
leg1, = plt.plot(cirf_time,cirf_raw,label = 'Empirical CIRF')
leg2, = plt.plot(cirf_time,epsp,':', label = 'Manual fit')
popt,pcov = scipy.optimize.curve_fit(spt.generateEPSP,cirf_time,
                                     cirf_raw,p)
cirf_fit = spt.generateEPSP(cirf_time,popt[0],popt[1],popt[2],popt[3])
leg3, = plt.plot(cirf_time,cirf_fit,'--',label = 'Least-squares fit')
plt.legend(handles = [leg1, leg2, leg3], fontsize = 18)
plt.xlabel('Time (s)', fontsize = 20)
plt.ylabel('Normalized amplitude', fontsize = 20)
plt.title('CIRF for regression', fontsize = 22)
plt.box('off')
plt.tick_params(labelsize = 14)
print('Parameters \n' +  str(popt))

#%% Load ephys data
import FileTools as ft
pyDir = r'S:\Avinash\SPIM\Alx\8-9-2015_Alx RG x 939_4dpf\Ephys\Fish2'
pyFileName = ft.findAndSortFilesInDir(pyDir,ext = 'mat')[-1]

pyData_disk = h5py.File(os.path.join(pyDir,pyFileName))['data']
pyData = {}
pyData['stim'],pyData['swim'] = {},{}
pyData['stim']['amps'] = pyData_disk['stim']['amps'][:]
pyData['stim']['inds'] = pyData_disk['stim']['inds'][:].astype(int)
pyData['swim']['startInds'] = pyData_disk['swim']['startInds'][:].astype(int)
pyData['swim']['distFromLastStim'] = pyData_disk['swim']['distFromLastStim'][:]
pyData['time'] = pyData_disk['t'][:]
pyData['smooth'] = np.transpose(pyData_disk['smooth']['burst'][:])

pyData['samplingRate'] = np.int(1/(pyData['time'][1]-pyData['time'][0]))



#%% Geting indices for stacks in ephys data and stims in optical data (can take upto 20 mins)
tic = time.time()
maxInt = int(20e-3*pyData['samplingRate'])
print('Geting indices for stacks in ephys data and stims in optical data...')
data['inds']['stim'] = spt.nearestMatchingInds(pyData['time'][pyData['stim']['inds'].astype(int)],data['time'])
data['inds']['stackInit'] = spt.nearestMatchingInds(data['time'],pyData['time'],processing = 'parallel')
escapeInds = pyData['swim']['startInds'][pyData['swim']['distFromLastStim'] < maxInt][0:-1]
data['inds']['escape'] = spt.nearestMatchingInds(pyData['time'][escapeInds],data['time'])
print(int(time.time()-tic),'sec')

#%% Subsampling ePhys signals to speed up subsequent regressor convoltion and also obtaining some useful indices
tic = time.time()
lpf = 100 # Low pass filter value
stimChan = 1

# Subsample(to speed up with convolution with Ca kernel) smoothed ePhys traces and time vector at about 4 times the Nyquist value
data['ephys'] = {}
dt_sub = int(pyData['samplingRate']/(lpf*4))
data['ephys']['smooth'] = spt.chebFilt(pyData['smooth'],1/pyData['samplingRate'],lpf,
                      btype = 'low')[::dt_sub]

time_sub = pyData['time'][::dt_sub]
samplingRate_sub = int(1/(time_sub[1]-time_sub[0]))

cirf_time_ephys = np.arange(cirf_time[0], cirf_time[-1],1/(samplingRate_sub))
mult = (cirf_time[1]-cirf_time[0])/(cirf_time_ephys[1]-cirf_time_ephys[0])
cirf_fit_ephys = spt.generateEPSP(cirf_time_ephys, popt[0], popt[1], popt[2], popt[3]*mult)

inds, thr = {},{}
blah = np.sum(data['ephys']['smooth'],axis = 1)
print('Obtaining thresholds for burst and escape detection...')
thr['burst'] = np.ceil(volt.getGlobalThr(blah))
inds['burst']={}
inds['burst']['comb'],amps = spt.findPeaks(blah, thr = thr['burst'])

foo = volt.getGlobalThr(amps)
foo2 = volt.getGlobalThr(np.delete(amps,np.where(amps>foo)))
thr['escape'] = 0.5*(foo + foo2)

# Plot escape threshold atop combined smooth traces as a check
plt.figure(figsize = (16,10))
plt.plot(time_sub,blah)
plt.axhline(y = thr['escape'], ls = '--')
plt.xlim(time_sub.min(),time_sub.max())
plt.tick_params(labelsize = 14)
plt.xlabel('Time (sec)', fontsize = 14)
plt.ylabel('Amplitude', fontsize = 14)
plt.title('Smoothed swim with overlaid escape threshold', fontsize = 14)
plt.box('off')

# Find escape indices
pyData['stim']['interval'] = np.floor(np.min(np.diff(pyData['time'][pyData['stim']['inds']].ravel()))).astype(int)
#inds['escape'] = spt.findPeaks(blah,thr = thr['escape'])[0]
inds['stim'] = spt.nearestMatchingInds(pyData['time'][pyData['stim']['inds']],time_sub)


# Mark escapes on previous plot
#plt.plot(time_sub[inds['escape']],blah[inds['escape']],'o', mfc = 'none',ms = 5)

stimVec = np.zeros(np.shape(time_sub))
stimVec[inds['stim']]=1

# Check for correspondence between stim times in different sampling space
plt.figure(figsize = (16,8))
plt.plot(pyData['time'].ravel()[pyData['stim']['inds'].ravel()]-time_sub[inds['stim']].ravel(),'.')
plt.axhline(y = 0, ls = '--', color = 'g')
plt.axhline(y = 2e-3, ls = '--')
plt.axhline(y = -2e-3, ls = '--')
#plt.xlim(time_sub.min(),time_sub.max())
plt.tick_params(labelsize = 14)
plt.xlabel('Index #',fontsize = 14)
plt.ylabel('Residue (sec)', fontsize = 14)
plt.box('off')
plt.title('Correspondence of stim times in high and low ePhys sampling space',fontsize = 14)

#inds['swim'] = spt.findPeaks(np.sum(data['ephys']['smooth'],axis =1), thr = thr['burst'],minPkDist = (1/5)*samplingRate_sub)[0]

# pkInds = [spt.findPeaks(signal,thr = volt.getGlobalThr(signal))[0] for signal in np.transpose(data['ephys']['smooth'])]

          
maxInt = np.int(30e-3 * samplingRate_sub) # Max interval after stim for a response to be seen as escape
nPreInds = 1
nPostInds = np.int(1.0*samplingRate_sub)
totLen = nPostInds + nPreInds

escapeCutter = np.zeros(np.shape(time_sub))

for escapeInd in inds['stim']:
    keepInds = np.arange(escapeInd-nPreInds,escapeInd+nPostInds)   
    if len(keepInds)>= totLen:
        keepInds = keepInds.astype(int)
        escapeCutter[keepInds]=1            
escapeCutter = escapeCutter.ravel()


#%% Getting light sheet stack indices in subsampled ePhys time space
# Getting light sheet stack indices in subsampled ePhys time space, and making regressors
inds['stackInit'] = spt.nearestMatchingInds(pyData['time'][data['inds']['stackInit']],time_sub,
                                            processing = 'parallel')
tic = time.time()        
print('Making regressors...')
X = {}
X['left'],X['right'] = {},{}
X['left']['escape'] = np.convolve(data['ephys']['smooth'][:,0]*escapeCutter,cirf_fit_ephys, mode = 'same')[inds['stackInit']]
X['right']['escape'] = np.convolve(data['ephys']['smooth'][:,1]*escapeCutter,cirf_fit_ephys, mode = 'same')[inds['stackInit']]
print(int(time.time()-tic),'sec')
X['left']['slow'] = np.convolve(data['ephys']['smooth'][:,0]*(1-escapeCutter),cirf_fit_ephys, mode = 'same')[inds['stackInit']]
X['right']['slow'] =np.convolve(data['ephys']['smooth'][:,1]*(1-escapeCutter),cirf_fit_ephys, mode = 'same')[inds['stackInit']]
print(int(time.time()-tic),'sec')
X['stim'] = np.zeros(np.shape(time_sub))
for ampNum, amp in enumerate(pyData['stim']['amps'][:,1]):
    X['stim'][inds['stim'][ampNum]] = amp
X['stim'] = np.convolve(X['stim'].ravel(),cirf_fit_ephys,mode = 'same')[inds['stackInit']]

print(int(time.time()-tic),'sec')
print(time.ctime())


#%% Plotting regressors
plt.figure(figsize = (16,10))
leg = list(np.zeros((5,1)))
leg[0], = plt.plot(data['time'],spt.standardize(X['left']['escape']), label = 'Left escape')
leg[1], = plt.plot(data['time'],-spt.standardize(X['right']['escape']),label = 'Right escape (sign inv)')
leg[2], = plt.plot(data['time'],spt.standardize(X['stim'])-2, label = 'Stim')
leg[3], = plt.plot(data['time'],spt.standardize(X['left']['slow'])-3, label = 'Left slow')
leg[4], = plt.plot(data['time'],-spt.standardize(X['right']['slow'])-3,label = 'Right slow (sign inv)')
plt.legend(handles = leg, fontsize = 13, loc= 1)
plt.xlabel('Time (sec)', fontsize  = 16)
plt.ylabel('Max-norm amplitude',fontsize = 16)
plt.yticks([])
plt.xlim(data['time'].min(),data['time'].max())
plt.title('Convolved regressors', fontsize = 18)
ax1 = plt.gca()
ax1.tick_params(labelsize = 14)
plt.box('off')

#%% Actual least squares linear regression
from sklearn import linear_model
#X['mat'] = np.array([X['stim'], X['left']['escape'], X['right']['escape'], X['left']['slow'] ,X['right']['slow']]).transpose()
X['mat'] = np.array([X['stim'], X['right']['escape'] ,X['right']['slow']]).transpose()
Y = data['kmeans']['X'].transpose()
regr = linear_model.LinearRegression()
regr.fit(X['mat'],Y)
Y_est = regr.predict(X['mat'])
sse = np.sum((Y-Y_est)**2,axis = 0)
Y_mean = np.tile(np.mean(Y,axis = 0),[np.shape(Y)[0],1])
ssto = np.sum((Y-Y_mean)**2,axis =0)
Rsq = 1-(sse/ssto)
Rsq_norm = spt.standardize(Rsq)*0.9 + 0.1
 
cmap = plt.cm.get_cmap('hsv')
B_pos,B_neg = regr.coef_.copy(), regr.coef_.copy()
B_pos[np.where(B_pos<0)]=0
B_neg[np.where(B_neg>0)]=0
B_neg =np.abs(B_neg)
cPos = np.tile(np.linspace(0,255,np.shape(X['mat'])[1]),[np.shape(regr.coef_)[0],1])
B_sum = np.sum(regr.coef_,axis = 1)
clr_pos = cmap(np.sum(B_pos*cPos,axis = 1)/B_sum)
clr_pos[:,3] = Rsq_norm

clr_neg = cmap(np.sum(B_neg*cPos,axis = 1)/B_sum)
clr_neg[:,3] = Rsq_norm

#%% Update data variable and save
data['inds']['stackInit'] = inds['stackInit']
data['pyData'] = pyData
data['regression'] ={}
data['regression']['X'] = X
data['regression']['Y'] = Y
data['regression']['Y_est'] = Y_est
data['regression']['B'] = regr.coef_.copy()
data['regression']['sse'] = sse
data['regression']['ssto'] = ssto
data['regression']['Rsq'] = Rsq
data['regression']['Rsq_norm'] = Rsq_norm
data['regression']['clr_pos'] = clr_pos
data['regression']['clr_neg'] = clr_neg
data['regression']['cirf'] = cirf_fit_ephys
 
del(X,Y,Y_est,Y_mean,B_sum,cPos,clr_pos, clr_neg, pyData)

#%% Plotting regression map - Method 1
imgStack_norm = spt.standardize(data['avg'].copy())
imgStack = np.transpose(np.tile(np.zeros(np.shape(imgStack_norm)),[4,1,1,1]),[1,2,3,0])
sliceList = np.arange(np.shape(data['avg'])[0])
alphaSlice = np.ones((np.shape(imgStack)[1],np.shape(imgStack)[2]))
for z in sliceList:   
    inds_cellsInSlice = np.where(data['info']['slice'].astype(int)==z+1)[0]
    imgStack[z,:,:,0:3] = volt.gray2rgb(imgStack_norm[z])
    imgStack[z,:,:,-1]= alphaSlice
    inds_cellsInSliceAndMask = np.intersect1d(inds_cellsInSlice,data['inds']['inMask'])
    for cInd in inds_cellsInSliceAndMask:
        cInd2 = np.where(data['inds']['inMask']==cInd)[0]
        rgba = data['regression']['clr_pos'][cInd2,:][0]
        pxls_cell = data['info']['inds'][cInd].ravel().astype(int)
        for pxl in pxls_cell:            
            pxl = np.unravel_index(pxl,np.shape(data['avg'][0].transpose()))
            for valNum,val in enumerate(rgba):
                imgStack[z,pxl[1],pxl[0],valNum] = val
imgStack = np.transpose(imgStack,[0,2,1,3])
   
plt.figure(figsize =(16,10))
plt.axis('auto')
plt.imshow(imgStack[20]);

           
           
sys.exit()
        
#%% Regressors in optical time

tic = time.time()
maxInt = np.int(30e-3 * pyData['samplingRate']) # Max interval after stim for response to be seen as escape
nPreInds = 1
nPostInds = np.ceil(1.5*(data['time'][1]-data['time'][0])).astype(int)
totLen = nPostInds + nPreInds

escapeCutter = np.zeros(np.shape(data['time']))

for escapeInd in data['inds']['stim']: # Changed from data['inds']['escape'] because many missing
    inds = np.arange(escapeInd-nPreInds,escapeInd+nPostInds)
    np.shape(inds)
    if len(inds)>= totLen:
        inds = inds.astype(int)
        escapeCutter[inds]=1   
        
print('Making regressors...')
X = {}
X['left'],X['right'] = {},{}
X['left']['escape'] = np.convolve((pyData['smooth'][data['inds']['stackInit'],0]*
                                   escapeCutter[:,0]), cirf_fit, mode = 'same')
X['right']['escape'] = np.convolve((pyData['smooth'][data['inds']['stackInit'],1]*
                                    escapeCutter[:,0]), cirf_fit,mode= 'same')
X['left']['slow'] = np.convolve((pyData['smooth'][data['inds']['stackInit'],0]*
                                 (1-escapeCutter[:,0])), cirf_fit, mode = 'same')
X['right']['slow'] = np.convolve((pyData['smooth'][data['inds']['stackInit'],1]*
                                  (1-escapeCutter[:,0])), cirf_fit, mode = 'same')

X['stim'] = np.zeros(np.shape(data['time']))[:,0]
for ind, amp in enumerate(pyData['stim']['amps'][:,1]):
    X['stim'][data['inds']['stim'][ind]] = amp
X['stim'] = np.convolve(X['stim'],cirf_fit,mode = 'same')
    
print(int(time.time()-tic),'sec')
print(time.ctime())

#%% Plotting regressors
plt.style.use('dark_background')
plt.figure(figsize = (16,10))
leg = list(np.zeros((5,1)))
leg[0], = plt.plot(data['time'],spt.standardize(X['left']['escape']), label = 'Left escape')
leg[1], = plt.plot(data['time'],-spt.standardize(X['right']['escape']),label = 'Right escape (sign inv)')
leg[2], = plt.plot(data['time'],spt.standardize(X['stim'])-2, label = 'Stim')
leg[3], = plt.plot(data['time'],spt.standardize(X['left']['slow'])-3, label = 'Left slow')
leg[4], = plt.plot(data['time'],-spt.standardize(X['right']['slow'])-3,label = 'Right slow (sign inv)')
plt.legend(handles = leg, fontsize = 13, loc= 1)
plt.xlabel('Time (sec)', fontsize  = 16)
plt.ylabel('Max-norm amplitude',fontsize = 16)
plt.yticks([])
plt.xlim(data['time'].min(),data['time'].max())
plt.title('Convolved regressors', fontsize = 18)
ax1 = plt.gca()
ax1.tick_params(labelsize = 14)
plt.box('off')