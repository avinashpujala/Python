
# coding: utf-8

# In[110]:


import numpy as np
import os, sys, time
import importlib
import matplotlib.pyplot as plt

codeDir = r'C:\Users\pujalaa\Documents\Code\Python\code'
sys.path.append(codeDir)
import apCode.AnalyzeEphysData as aed
import apCode.SignalProcessingTools as spt



#%% Inputs
epDir         = r'S:\Avinash\Ablations and behavior\NV paper\20180509\f3'
preFile       = r'preAblation_01.10chFlt'
postFile      = r'postAblation_20minsLater.10chFlt'

stimCh        = 0
preStimPer    = 0.1 # In sec
postStimPer   = 2
Fs            = 6000 # Sampling rate

stimCh = 'stim'+ str(stimCh)
dt = 1/Fs
preStimPts = preStimPer*Fs
postStimPts = postStimPer*Fs


# In[112]:


importlib.reload(aed)
pre = aed.import10ch(os.path.join(epDir,preFile))
post = aed.import10ch(os.path.join(epDir,postFile))
print(pre.keys())
print(time.ctime())


# In[155]:


#%% Detect and check stimulus indices
pre['stimInds'] = spt.findPeaks(spt.zscore(pre[stimCh]),thr=3)[0]-2
post['stimInds'] = spt.findPeaks(spt.zscore(post[stimCh]),thr=3)[0]-2

xmin = np.max((pre['t'][0],post['t'][0]))
xmax = np.min((pre['t'][-1],post['t'][-1]))
plt.style.use(('seaborn-dark','seaborn-colorblind','seaborn-poster'))
plt.subplot(2,1,1)
plt.plot(pre['t'],pre[stimCh])
plt.plot(pre['t'][pre['stimInds']],pre[stimCh][pre['stimInds']],'o',markersize =15)
plt.xlim(xmin,xmax)
plt.title('Pre-Ablation')

plt.subplot(2,1,2)
plt.plot(post['t'],post[stimCh])
plt.plot(post['t'][post['stimInds']],post[stimCh][post['stimInds']],'o',markersize =15)                                 
plt.xlim(xmin,xmax)
plt.xlabel('Time (s)')
plt.ylabel('Stim Amplitude')
plt.title('Post-Ablation')
plt.suptitle('Stimulus detection',fontsize = 18);


# In[159]:


X = {}
X['pre'] = np.array((pre['t'],pre['ch0'],pre['ch1']))
X['post'] = np.array((post['t'],post['ch0'],post['ch1']))


# In[169]:


pre['trl'] = aed.segmentByEvents(X['pre'],pre['stimInds'],preStimPts,postStimPts)


# In[171]:


postStimPts


# In[167]:


pre['stimInds']


# In[34]:





# In[53]:


fig = plt.figure(figsize=[20,10])

start = 10000
stop = start + 30000
step = 10
plrange = np.arange(start,stop,step)

plt.subplot(2,1,1)
plt.plot(plrange,dat1[0,plrange])
plt.title('spim rig whitened signal, ch0')

plt.subplot(2,1,2)
plt.plot(plrange,dat2[0,plrange],'k')
plt.title('fictive behavior rig whitened signal, ch0')


# In[54]:


plt.figure(figsize = (10,10))

plt.subplot(2,1,1)
plt.hist(dat1[0,10000:],200);
plt.title('spim rig whitened histogram' )

plt.subplot(2,1,2)
plt.hist(dat2[0,10000:],200,color='k');
plt.title('fictive behavior rig whitened histogram' )


# In[49]:


#plt.plot(dat[6,0:5200000])
fig = plt.figure(figsize=[20,10])
start = 0
stop = dat.shape[1]
step = 100
plrange = np.arange(start,stop,step)
toPlot = [0,1,2]
nPlots = len(toPlot)
for p in range(nPlots):
    plt.subplot(nPlots,1,p+1)
    plt.plot(plrange,dat[toPlot[p],plrange],'r')


# In[46]:


range(nPlots)

