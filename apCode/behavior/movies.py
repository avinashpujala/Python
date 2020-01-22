# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:17:41 2019

@author: pujalaa
"""

def fishPos(images, pos = None, fps = 30, n_pts = 50, display = True, save = False, 
            savePath = None, **kwargs):
    """
    Makes a movie of fish images with overlaid fish position
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation
    plt.rcParams['animation.ffmpeg_path'] = r'V:\Code\Python\FFMPEG\bin\ffmpeg.exe'
    from IPython.display import HTML
    import time
    
    nImages = images.shape[0]
    if np.all(pos == None):
        pos = np.zeros((nImages,2))*np.nan
    cmap = kwargs.get('cmap', 'gray')
    interp = kwargs.get('interpolation', 'nearest')
    dpi = kwargs.get('dpi', 30)
    plt.style.use(('seaborn-poster', 'dark_background'))
    fh = plt.figure(dpi = dpi, facecolor='k', figsize =(14,14))
    ax = fh.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    cLim = images.min(), images.max()
    mov = ax.imshow(images[0], cmap= cmap, interpolation = interp, 
                    vmin = cLim[0], vmax = cLim[1])  
    
    pos_extra = np.concatenate((np.zeros((n_pts,2))*np.nan,pos))
    ax.plot(pos_extra[:n_pts,0], pos_extra[:n_pts,1],'r.', ms= 1)   
    
    def update_img(n):        
        mov.set_data(images[n])   
        ax.set_title('Frame # {}'.format(n))
        ax.plot(pos_extra[n:n_pts+n,0], pos_extra[n:n_pts+n,1],'r.',ms = 1)
        
    ani = animation.FuncAnimation(fh,update_img,np.arange(nImages),
                                  interval= 1000/fps, repeat = False)    
    plt.close(fh)
    
    if save:
        print('Saving...')
        writer = animation.writers['ffmpeg'](fps=fps)
        if savePath != None:
            ani.save(savePath, writer = writer, dpi = dpi)
            print('Saved to \n{}'.format(savePath))
        else:
            vidName = 'video_{}.mp4'.format(time.strftime('%Y%m%d'))
            ani.save(vidName, writer=writer, dpi=dpi)
            print('Saved in current drirve as \n{}'.format(vidName))
        
    if display:
        print('Displaying...')
        return HTML(ani.to_html5_video())
    else:
        return ani 