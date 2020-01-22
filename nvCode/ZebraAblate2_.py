# tool for circular ROI selection and saving in 2P excitation profile
# by Nikita Vladimirov <nvladimus@gmail.com>
# started 08Jan2015
from ij import IJ
from ij.plugin.frame import RoiManager
from java.awt.event import MouseAdapter, KeyEvent, KeyAdapter
from ij.gui import GenericDialog, WaitForUserDialog, GenericDialog, Roi, Overlay
from ij.io import SaveDialog

def getOptions():
 global listener, xlist, ylist, zlist, manager
 gd = GenericDialog("Target Selection")
 gd.addChoice('type', ['point', 'circle', 'spiral'], 'point')
 gd.addNumericField("                power (%)", 85, 0)
 gd.addNumericField("                duration (ms)", 1, 0) 
 gd.addNumericField("                radius(circle/spiral)", 5, 0) 
 gd.addNumericField("                # revolutions (circle/spiral)", 3, 0)
 gd.addNumericField("                add CAMERA_TRIGGER after every (entries)", 1, 0)
 gd.addNumericField("                Prewait between entries (ms)", 5000, 0)
 gd.addNumericField("                Z-start of stack (um)", 0, 0)
 gd.addNumericField("                Z-step of stack (um)", 5, 0)
 gd.addMessage('Press ENTER to save\n')
 gd.addMessage('Press ESC to restart\n')
 gd.showDialog()
 profileType = gd.getNextChoice()
 power = gd.getNextNumber()
 duration = gd.getNextNumber() 
 r = gd.getNextNumber()
 Nturns = gd.getNextNumber()
 camTriggerEvery = gd.getNextNumber()
 prewait = gd.getNextNumber()
 zStart = gd.getNextNumber()
 zStep = gd.getNextNumber()
 if gd.wasCanceled():
    IJ.setTool(Toolbar.RECTANGLE)
    return 
 else: return r, power, profileType, duration, Nturns, camTriggerEvery, zStart, zStep, prewait

def reset():
 global radius, iROI, power, profileType, duration, Nturns, xlist, ylist, zlist, camTriggerEvery, zStart, zStep, prewait
 xlist = []
 ylist = []
 zlist = []
 manager.runCommand('Reset')
 manager.runCommand('Show All')
 iROI = 0
 options = getOptions()
 if options is not None:
    radius, power, profileType, duration, Nturns, camTriggerEvery, zStart, zStep, prewait = options
 
class ML(MouseAdapter):
 def mousePressed(self, keyEvent):
  global iROI, xlist, ylist, zlist
  iROI += 1
  canv = imp.getCanvas()
  p = canv.getCursorLoc()
  z = imp.getSlice()
  roi = OvalRoi(p.x - radius, p.y - radius, radius*2, radius*2)
  roi.setName('z' + str(z) + 'cell' + str(iROI))
  roi.setPosition(z)
  xlist.append(p.x)
  ylist.append(p.y)
  zlist.append(z)
  imp.setRoi(roi)
  manager.addRoi(roi)
  manager.runCommand('Draw')

class ListenToKey(KeyAdapter):
  def keyPressed(this, event):
    doSomething(event)

def doSomething(keyEvent):
  """ A function to react to key being pressed on an image canvas. """
  global iROI, xlist, ylist, zlist, power, profileType, duration, Nturns, hOffset, camTriggerEvery, zStart, zStep, prewait
  print "clicked keyCode " + str(keyEvent.getKeyCode())
  if keyEvent.getKeyCode() == 10: # Enter is pressed!
      sd = SaveDialog('Save ROIs','.','Eprofile','.txt')
      directory = sd.getDirectory()
      filename = sd.getFileName()
      filepath = directory + filename
      print filepath
      f = open(filepath, 'w')
      for i in range(len(xlist)):
            f.write('ENTRY_START\n')
            f.write('ABLATION(OFF,200.0)\n')
            f.write('PRE_WAIT(' + str(prewait)+ ')\n')
            f.write('PRE_TRIGGER(OFF,5000,CONTINUE)\n')
            f.write('COORDINATES('+str(xlist[i])+','+str(hOffset + ylist[i])+','+str(zStart + (zlist[i]-1)*zStep)+')\n')
            if(profileType == 'point'):
                f.write('SCAN_TYPE(POINT)\n')
            if(profileType == 'circle'):
                f.write('SCAN_TYPE(CIRCLE,'+ str(radius) + ',' + str(round(duration/Nturns)) +')\n')
            if(profileType == 'spiral'):
                f.write('SCAN_TYPE(SPIRAL,' + str(radius) + ',' + str(duration) + ',0,'+str(Nturns)+')\n')     
            f.write('POWER(' + str(power) + ')\n')
            f.write('DURATION(' +str(duration)+')\n')
            f.write('POST_TRIGGER(OFF,0, CONTINUE)\n')
            f.write('CAMERA_TRIGGER\n')
            f.write('ENTRY_END\n\n')
      f.close()
      manager.runCommand('Save',directory+filename+'.zip')
  if keyEvent.getKeyCode() == 27: # Esc is pressed!
      reset()
  #if keyEvent.getKeyCode() == 127: # DEL is pressed!
  #    killIt = true    
  #    return  
  # Prevent further propagation of the key event:
  keyEvent.consume()

# MAIN code         
imp = IJ.getImage()
killIt = False
if imp.getHeight() == 1024: # half-chip size
    hOffset = 512
elif imp.getHeight() == 2048: #full-chip image
    hOffset = 0
else:
    gdGoodBye = GenericDialog("Dimensions error")
    gdGoodBye.addMessage('Image must be 2048x2048, or 2048x1024 pixels! \n I retire!')
    gdGoodBye.showDialog()
    killIt = True

if not killIt:
    IJ.setTool(Toolbar.RECTANGLE)
    manager = RoiManager.getInstance()
    if manager is None:
        manager = RoiManager();   
    reset()
    listener = ML()
    keyLis = ListenToKey()
    win = imp.getWindow()
    win.getCanvas().addMouseListener(listener)
    win.getCanvas().addKeyListener(keyLis)



