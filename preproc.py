import glob, re, os
import numpy as np
from scipy import signal,fft
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
from neurodot.core.data_processing.session_data import SessionData
import tables
from tables import NoSuchNodeError

#-------------------------------------------------------------------------------
# wavelet smoothing
import pywt
from statsmodels.robust import mad
from copy import deepcopy

def waveletSmooth( x, wavelet="coif3", level=1):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    # calculate a threshold
    sigma = mad( coeff[-level] )
    # changing this threshold also changes the behavior,
    # but I have not played with this very much
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    filt_coeff = deepcopy(coeff)
    filt_coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec( filt_coeff, wavelet, mode="per" )
    return y, coeff
#-------------------------------------------------------------------------------

#set the current directory as the default path for saving figures
import matplotlib
matplotlib.rcParams['savefig.directory'] = '.'

STANDARD_FONTSIZE = 20
matplotlib.rcParams.update({'font.size': STANDARD_FONTSIZE})

COLOR_CYCLE = ['r','g','b','c','m','k','y',(0.5,0.5,0.5)]
plt.rc('axes', color_cycle=COLOR_CYCLE)


#*******************************************************************************
# match the data set
DATE_STRING = "2017-08-09"
SESSION_NUMBER = 1
SUBJECT_NAME = "CWV"
RECORDING_NUM = 0

POLARITY = -1.0 #for the new board revF
CHANNELS = [1,2,3,4,5,6,7,8]
O1_channels = [1,2,3,4]
Oz_channels = [2,4,5,7]
O2_channels = [5,6,7,8]



DATAFILE_PATTERN = "%s*SN%03d*.h5" % (DATE_STRING,SESSION_NUMBER)
print("Matching pattern: %s" % DATAFILE_PATTERN)
DATAFILES = glob.glob(DATAFILE_PATTERN)
DATAFILES.sort()
DATAFILE = DATAFILES[0]
print DATAFILE

_, datafilename = os.path.split(DATAFILE)
datafilename, _ = os.path.splitext(datafilename)



#*******************************************************************************
#load the data
sd = SessionData.load(DATAFILE) #requires neurodot package

#grab the ads data from the first recording
rec = sd.set_recording(RECORDING_NUM)
rmd = sd.get_recording_metadata()
sample_rate = rmd['sample_rate']
print sample_rate

D = rec.ads_samples.read()
t = D['ts']
t0 = t[0]

#*******************************************************************************
#compute filter coefficients
F_NYQ = sample_rate/2.0
BP_lc = 1.0#Hz
BP_hc = 100.0
BP_taps = 4
BP_b, BP_a = signal.bessel(BP_taps, [BP_lc/F_NYQ, BP_hc/F_NYQ],'bandpass')

BS30_lc = 28.0 #Hz
BS30_hc = 32.0
BS30_taps = 5
BS30_b, BS30_a = signal.butter(BS30_taps, [BS30_lc/F_NYQ, BS30_hc/F_NYQ],'bandstop')


BS50_lc = 50.0 #Hz
BS50_hc = 70.0
BS50_taps = 5
BS50_b, BS50_a = signal.butter(BS50_taps, [BS50_lc/F_NYQ, BS50_hc/F_NYQ],'bandstop')


BS60_lc = 58.0 #Hz
BS60_hc = 62.0
BS60_taps = 5
BS60_b, BS60_a = signal.butter(BS60_taps, [BS60_lc/F_NYQ, BS60_hc/F_NYQ],'bandstop')

BS120_lc = 119 #Hz
BS120_hc = 121
BS120_taps = 2
BS120_b, BS120_a = signal.butter(BS120_taps, [BS120_lc/F_NYQ, BS120_hc/F_NYQ],'bandstop')

#filter the potentials
Vs = []
for i,chan in enumerate(CHANNELS):
    col_name = "V%03d" % chan
    V_raw = D[col_name]  #pull out each channel's trace
    V = V_raw
    #apply bandpass filter
    V = signal.filtfilt(BP_b,BP_a,V)
    #apply bandstop filters
    #V = signal.filtfilt(BS50_b,BS50_a,V)
    #V = signal.filtfilt(BS30_b,BS30_a,V)
    V = signal.filtfilt(BS60_b,BS60_a,V)
    V = signal.filtfilt(BS120_b,BS120_a,V)
    V *= POLARITY
    Vs.append(V)
Vs = np.array(Vs)

#*******************************************************************************

#*******************************************************************************
#partion data into epochs
EPOCH_FLAGS = [1,2,3,4]
BLOCK_FLAGS = {
    11:1,
    12:2,
    13:3,
    14:4,
}

FIX_BLOCK_FLAGS = True
COMBINE_EPOCHS  = False
LEFT_RIGHT_TIMING_CORRECTION = 0.0025  #this is emperical

#constrain the epoch length between these limits
n_left  = int(sample_rate*0.0)
n_right = int(sample_rate*0.4)
i_max   = len(V) - 1

epochs = dict([(k,[]) for k in EPOCH_FLAGS])
block_flag  = None
epoch_flag  = 0
epoch_start = 0
epoch_end   = 0
i = 0
for event in rec.vsync_events.iterrows():
    ts   = event['ts']
    flag = event['flag']
    try:
        if flag in BLOCK_FLAGS: #start of new stimuli block
            block_flag = flag
            print "Detected block flag:", flag
        elif flag in EPOCH_FLAGS:
            epoch_flag = flag
            print ts, flag, t[i], epoch_end - epoch_start
            #fix broken flags
            if flag == BLOCK_FLAGS.get(block_flag):
                pass #ok no fix needed
            elif block_flag is None:
                print "Warning: no, block detected yet, SKIPPING epoch"
                epoch_flag = None
            else:
                print "Warning: flag '%d' doesn't match block_flag '%s - 10'" % (flag,block_flag)
                if FIX_BLOCK_FLAGS:
                    bf = BLOCK_FLAGS.get(block_flag)
                    if bf is None:
                        raise ValueError("cannot fix block with block_flag = %s" % block_flag)
                    print "\tFIXING to flag: %d" % bf
                    epoch_flag = bf
                else:
                    print "\tSKIPPING epoch"
                    epoch_flag = None
            #save the epoch
            if not epoch_flag is None:
                #adjust for left vs right eye timing
                if not LEFT_RIGHT_TIMING_CORRECTION is None:
                    if epoch_flag in (1,3):
                        ts -= LEFT_RIGHT_TIMING_CORRECTION
                    elif epoch_flag in (2,4):
                        ts += LEFT_RIGHT_TIMING_CORRECTION
                while t[i] < ts:
                    i += 1
                epoch_start = i + n_left
                epoch_end   = i + n_right
                #combine epochs
                if COMBINE_EPOCHS:
                    epoch_flag = 1
                if epoch_end < i_max:
                    l = epochs[epoch_flag]
                    l.append((epoch_start,epoch_end))
                else:
                    print "Warning: epoch extends out of bounds, SKIPPING"
    except IndexError as exc:
        print("Warning caught exception: %s" % exc)


#*******************************************************************************
# generate figure
peak_P1 = {
    
}

peak_N1 = {
    "search_left_P1": (10,70),
}


PEAKS1 = {
    'P1':{"range": (80,170),},
    'N1' :{
            "search_left_P1": (30,70),
            #'fix_x': 68.0,
            },
}

PEAKS2 = {
    'P1':{"range": (80,170),},
    'N1' :{
            "search_left_P1": (30,70),
            #'fix_x': 73.0,
            },
}

PEAKS3 = {
    'P1':{"range": (80,170),},
    'N1' :{"search_left_P1": (30,70),},
}

PEAKS4 = {
    'P1':{"range": (80,170),},
    'N1' :{"search_left_P1": (30,70),},
}




USE_STD_BANDS = False#True

FIGSIZE = (8.5*1.5,11.0*1.5)
EPOCH_FLAG_SETS = [(1,2),(3,4)]
LABELS = ["High Contrast","Low Contrast"]
DATASET_NAMES = [("L-HC","R-HC"),("L-LC","R-LC")]

fig1 = plt.figure(figsize=FIGSIZE)
#set up axes
ax1_1 = fig1.add_subplot(211) #EEG
#ax1r = ax1.twinx()
ax1_2 = fig1.add_subplot(212) #EEG

fig2 = plt.figure(figsize=FIGSIZE)
#set up axes
ax2_1 = fig2.add_subplot(211) #EEG
#ax2r = ax1.twinx()
ax2_2 = fig2.add_subplot(212) #EEG

FIGURES   = [fig1,fig2]
AXES_SETS = [(ax1_1,ax1_2),(ax2_1,ax2_2)]
PEAK_SETS = [(PEAKS1,PEAKS2),(PEAKS3,PEAKS4)]

LOCATIONS = ['O1','Oz','O2']
CHANNEL_SETS = [O1_channels,Oz_channels,O2_channels]
COLORS = ["b","k","r"]


LINESTYLES = ["-"]

#XLIM = (-125.0,125.0)
YLIM = (-10.5,10.5)
V_UNITS = 1e-6 #microvolts

#epoch filtering parameters
V_MAX_AMP = 30*1e-6


#FFT params
N_PERSEG = 2**13
N_FFT = 2**13
fr = None  #will hold real valued frequency domain x axis
fc = None  #will hold complex valued frequency domain x axis

tVEP_data = {}

for epoch_flags, label, dsnames, axes, peak_set in zip(EPOCH_FLAG_SETS, LABELS, DATASET_NAMES, AXES_SETS, PEAK_SETS):
    print "--------------------------------------------------------------"
    print "epoch_flags:", epoch_flags
    print "label:", label
    print "axes:", axes
    ax1, ax2 = axes
    for n, ax, dsname, peak_dict in zip(epoch_flags,axes, dsnames, peak_set):
        
        print n
        print "---"
        # 
        Vs_epochs     = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
        Vs_Pxx_epochs = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
        fr = None
        for i,chan in enumerate(CHANNELS):
            for start,end in epochs[n]:
                V = Vs[i]
                V_epoch = V[start:end]
                #V_epoch = signal.detrend(V_epoch, type="linear")
                #V_epoch -= V_epoch[0]
                max_amp = abs(V_epoch).max()
                if (max_amp > -V_MAX_AMP) and (max_amp < V_MAX_AMP):
                    Vs_epochs[chan].append(V_epoch)
                    #fr, V_Pxx = signal.welch(V_epoch/V_UNITS,fs=sample_rate, nperseg=N_PERSEG, nfft=N_FFT)
    #                win_V_epoch = V_epoch
    #                win_V_epoch = np.hanning(len(V_epoch))*win_V_epoch
    #                fr, V_Pxx = signal.periodogram(win_V_epoch/V_UNITS,fs=sample_rate, nfft=N_FFT)
    #                Vs_Pxx_epochs[chan].append(V_Pxx)
                else:
                    print "WARNING: rejecting chan %d V_epoch with max_amp = %0.1f uV" % (chan, max_amp/1e-6)
        
        #---------------------------------------------------------------------------
        for location, channels, color in zip(LOCATIONS,CHANNEL_SETS,COLORS):
            V_epochs = []
            
            Vs_em_ALL = []
            Vs_em_ALL_std = []
            for chan in channels:
                #time domain
                V_epochs = Vs_epochs[chan]
                #perform wavelet smoothing on each epoch
                #V_epochs = [waveletSmooth(V_epoch, wavelet="coif3", level=5)[0] for V_epoch in V_epochs]
                #gather into one array
                V_epochs = np.vstack(V_epochs)
                print "len(V_epochs) =", len(V_epochs)
                V_epochs_mean   = V_epochs.mean(axis=0)
                Vs_em_ALL.append(V_epochs_mean)
                #V_epochs_median = np.median(V_epochs, axis=0)
                Vs_em_ALL_std.append(V_epochs.std(axis=0))
                
            Vs_em_ALL = np.vstack(Vs_em_ALL).mean(axis=0)
            Vs_em_ALL_std = np.vstack(Vs_em_ALL_std).mean(axis=0)
            
            # Time Domain plots
            t = 1e3*np.arange(Vs_em_ALL.shape[0])/float(sample_rate)
            y = Vs_em_ALL/V_UNITS
            N = len(V_epochs)
            #ymed = V_epochs_median/1e-6
            yerr =  Vs_em_ALL_std/V_UNITS/np.sqrt(N)
            yn = y - yerr
            yp = y + yerr
            ax.plot(t,y, label=location, color=color, linestyle="-")
            if USE_STD_BANDS:
                ax.fill_between(t,yn,yp, facecolor = color, alpha = 0.25)
                
            #save dataset
            dsfname = "%s-%s" % (dsname, location)
            tVEP_data[dsfname] = {'x':t, 'y':y,'yerr':yerr, 'Vs_epochs':Vs_epochs}
        
            #interpolate for peak fitting
            y_interp = interp1d(t,y,kind="cubic")

            #only fit peaks on Oz data
            if location == "Oz":
                #P1
                P1 = peak_dict.get('P1')
                if not P1 is None:
                    try:
                        x0,x1 = P1['range']
                        Xnew = np.linspace(x0,x1,1000)
                        Ynew = y_interp(Xnew)
                        i_p = Ynew.argmax()
                        while i_p == 0 or i_p == len(Xnew) - 1:
                            #continue searching right
                            Xnew = np.linspace(Xnew[0]+1,Xnew[-1]+1,1000)
                            Ynew = y_interp(Xnew)
                            i_p = Ynew.argmax()
                        y_p = Ynew[i_p]
                        x_p = Xnew[i_p]
                        P1['x'] = x_p
                        P1['y'] = y_p
                        ax.axvline(x = x_p, color=color, linestyle="--")
                    except Exception as err:
                        print("WARNING: in P1 peak fitting caught error: %s" % err)
                #N1
                N1 = None
                N1 = peak_dict.get('N1')
                if not N1 is None:
                    try:
                        x_p = None
                        y_p = None
                        if 'fix_x' in N1:
                            x_p = N1['fix_x']
                            Xnew = np.linspace(x_p-10,x_p + 10,101)
                            Ynew = y_interp(Xnew)
                            y_p = Ynew[50]
                        elif 'search_left_P1':
                            i_p = None
                            search_left_min, search_left_max = N1['search_left_P1']
                            for search_left in np.arange(search_left_min,search_left_max):
                                x0 = P1['x']-search_left
                                x1 = P1['x']-search_left_min
                                Xnew = np.linspace(x0,x1,1000)
                                Ynew = y_interp(Xnew)
                                i_p = Ynew.argmin()
                                print("x0:",x0,"x1:",x1,"i_p:",i_p)
                                if i_p > search_left_min//2:# and  Ynew[i_p] < 0.0:
                                    break #we've found a negative local minimum
                            y_p = Ynew[i_p]
                            x_p = Xnew[i_p]
                        N1['x'] = x_p
                        N1['y'] = y_p
                        ax.axvline(x = x_p, color=color, linestyle="--")
                        ax.annotate('N1 (%0.01f ms)' % (x_p,), 
                                    xy=(x_p, y_p), 
                                    xytext=(x_p + 5, y_p - 1.5),
                                    color=color,
                                    backgroundcolor=(1.0,1.0,1.0,0.9),
                                #arrowprops=dict(facecolor='black', shrink=0.05),
                                )
                        ax.plot([x_p], [y_p], marker='o', markersize=10, color=color)
                    except Exception as err:
                        print("WARNING: in N1 peak fitting caught error: %s" % err)
                #N2
                N2 = peak_dict.get('N2')
                if not 'N2' is None:
                    try:
                        i_p = None
                        Xnew = np.linspace(P1['x'] + 30,P1['x'] + 100,1000)
                        Ynew = y_interp(Xnew)
                        i_p = Ynew.argmin()
                        y_p = Ynew[i_p]
                        x_p = Xnew[i_p]
                        N2 = {'x':x_p,'y':y_p}
                        ax.axvline(x = x_p, color=color, linestyle="--")
                        ax.annotate('N2 (%0.01f ms)' % (x_p,), 
                                    xy=(x_p, y_p), 
                                    xytext=(x_p + 5, y_p - 1.5),
                                    color=color,
                                    backgroundcolor=(1.0,1.0,1.0,0.9),
                                #arrowprops=dict(facecolor='black', shrink=0.05),
                                )
                        ax.plot([x_p], [y_p], marker='o', markersize=10, color=color)
                    except Exception as err:
                        print("WARNING: in N2 peak fitting caught error: %s" % err)
                if not P1 is None:
                    if not N1 is None:
                        ax.annotate('P1 (%0.01f ms, %0.01f uV)' % (P1['x'], P1['y'] - N1['y']), 
                                        xy=(P1['x'], P1['y']), 
                                        xytext=(P1['x'] + 5, P1['y'] + 0.5),
                                        color=color,
                                        backgroundcolor=(1.0,1.0,1.0,0.9),
                                    #arrowprops=dict(facecolor='black', shrink=0.05),
                                    )
                        ax.plot([P1['x']], [P1['y']], marker='o', markersize=10, color=color)
                    else:
                        ax.annotate('P1 (%0.01f ms)' % (P1['x'],), 
                                        xy=(P1['x'], P1['y']), 
                                        xytext=(P1['x'] + 5, P1['y'] + 0.5),
                                        color=color,
                                        backgroundcolor=(1.0,1.0,1.0,0.9),
                                    #arrowprops=dict(facecolor='black', shrink=0.05),
                                    )
                        ax.plot([P1['x']], [P1['y']], marker='o', markersize=10, color=color)
    
    
#POST PLOT FORMATTING
for fig, axes, label in zip(FIGURES,AXES_SETS, LABELS):
    ax1, ax2 = axes
    title = "%s\nStimulus Condition '%s'" % (datafilename,label)
    fig.suptitle(title, fontsize = STANDARD_FONTSIZE - 2)

    ax1.set_title("Left Eye")
    ax1.legend(loc="upper right", fontsize = STANDARD_FONTSIZE - 2)
    #ax1r.legend(loc="upper right")
    #ax1_1.set_xlabel("Time [ms]")
    ax1.set_ylabel("EEG [$\mu$V]")
    ax1.set_ylim(YLIM)
    ax1.grid(True)

    ax2.set_title("Right Eye")
    ax2.legend(loc="upper right", fontsize = STANDARD_FONTSIZE - 2)
    #ax1r.legend(loc="upper right")
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("EEG [$\mu$V]")
    ax2.set_ylim(YLIM)
    ax2.grid(True)

    #    title = "%s\n%s" % (datafilename, "Averaged Spectrum")
    #    FD_f1ax1.set_title(title)
    #    FD_f1ax1.legend(loc="upper right")
    #    FD_f1ax1.set_xlabel("Frequency [Hz]")
    #    FD_f1ax1.set_ylabel(r"Power [$\left(\mu V\right)^2/$Hz]")

    #ax2_1.set_xlim(0.0,125.0)

    ##fig.savefig("EFEG_SSVEP_epoch_avg.png")
    #outdata = np.vstack(outdata).transpose()
    #np.savetxt(datafilename + ".csv", outdata, delimiter=",")
    
################################################################################
#figure derived from final data
#print tVEP_data



#fig3 = plt.figure(figsize=FIGSIZE)
##set up axes
#ax3_1 = fig3.add_subplot(211) #EEG
##ax1r = ax1.twinx()
#ax3_2 = fig3.add_subplot(212) #EEG

#ds1 = tVEP_data['L-HC-Oz']
#ds2 = tVEP_data['L-BG-Oz']

#y1 = ds1['y']
#y2 = ds2['y'] 

#ax3_1.plot(ds1['x'],y1, color="k", linestyle="-", label="L-HC-Oz")
#ax3_1.plot(ds1['x'],y2, color="k", linestyle="--", label="L-BG-Oz")

#for l in range(4,10):
#    print "level =",l
#    try:
#        y1ws, coeff1 = waveletSmooth(y1, level=l)
#        ax3_1.plot(ds1['x'],y1ws, 
#                   linestyle="-", 
#                   label="L-HC-Oz-WS%d" % l
#                   )
#        y2ws, coeff2 = waveletSmooth(y2, level=l)
#        ax3_1.plot(ds2['x'],y2ws, 
#                   linestyle="--", 
#                   label="L-BG-Oz-WS%d" % l
#                   )
#    except IndexError:
#        print "stopping"
#        break
#        
#ax3_1.set_ylim(YLIM)
#ax3_1.grid(True)
#ax3_1.legend(loc="upper right")

#coeff = coeff2
#for i,c in enumerate(coeff):
#    y = coeff[i]
#    x = np.arange(len(y))
#    ax3_2.bar(x,y, 
#              width=0.8,
#              color = COLOR_CYCLE[i],
#              edgecolor='None',
#              align='center',
#              label="%d" % i
#              )
#ax3_2.grid(True)
#ax3_2.legend(loc="upper right")
#               




#show the graphs but allow program to continue
plt.show(block = False)
#wait for the user
input("press enter to exit")
plt.close('all')
