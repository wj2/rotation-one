
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_matlab_hdf5(path, datasetpaths, iterref=True):
    repldict = {}
    with h5py.File(path, 'r') as f:
        for dsetpath in datasetpaths:
            dataset = np.array(f[dsetpath])
            if iterref:
                for i, item in enumerate(dataset):
                    derefd = np.array(f[item[0]])
                    dataset[i, 0] = derefd
            repldict[dsetpath] = dataset
    return repldict

def peak_to_trough(wvforms):
    ptt = np.empty(wvforms.shape[0])
    for i, wv in enumerate(wvforms):
        amin = wv[0].argmin()
        amax = wv[0].argmax()
        ptt[i] = amin - amax
    return np.abs(ptt)
               
def get_inhibexcit(rasts, conds, p2ts, inhib=10, excit=15):
    inhib_r = rasts[p2ts <= inhib]
    inhib_c = conds[p2ts <= inhib]
    excit_r = rasts[p2ts >= excit]
    excit_c = conds[p2ts >= excit]
    return inhib_r, inhib_c, excit_r, excit_c
 
def array_format(rasts, conds, cutoff, tail=30, buff=0, qual=None, thresh=None,
                 nstim=25, getgood=True):
    if getgood:
        rasts, conds = get_good_neurons(rasts, conds, qual, thresh)
    frs = np.empty(rasts.shape, dtype=object)
    nov = np.empty((rasts.shape[0], nstim, rasts[0].shape[0]))
    fam = np.empty((rasts.shape[0], nstim, rasts[0].shape[0]))
    for i, rast in enumerate(rasts):
        frs[i] = get_ifr_rastlist(rast)
        for stim in np.unique(conds[i]):
            inds = np.where(stim == conds[i])
            stim_frs = frs[i][:, inds[1]]
            stim_nov = stim_frs[:, :cutoff]
            stim_fam = stim_frs[:, cutoff+buff:cutoff+buff+tail]
            meanstim_nov = stim_nov.mean(1)
            meanstim_fam = stim_fam.mean(1)
            nov[i, stim-1, :] = meanstim_nov
            fam[i, stim-1, :] = meanstim_fam
    return nov, fam

def get_good_neurons(raster, conds, p2ts, qual, thresh=3):
    good_rasters = raster[qual >= thresh]
    good_conds = conds[qual >= thresh]
    good_p2ts = p2ts[qual >= thresh]
    return good_rasters, good_conds, good_p2ts

def get_rate_window(rasts, begin, end):
    rsum = 1000. * rasts[begin:end, :].sum(0) / (end - begin)
    return rsum

def get_ifr_rastlist(rasts):
    frs = np.empty(rasts.shape)
    for i in xrange(rasts.shape[1]):
        rast = rasts[:, i]
        fr = raster_to_ifr(rast)
        frs[:, i] = fr
    return frs

def get_ifr_window(rasts, begin, end):
    frs = []
    for i in xrange(rasts.shape[1]):
        rast = rasts[:, i]
        fr = raster_to_ifr(rast)
        frs.append(fr[begin:end].mean())
    return np.array(frs)

def mean_normalize(rates):
    return (rates - rates.mean()) / rates.std()
    
def max_normalize(rates):
    return rates / float(rates.max())

def get_avg_rates(rasters, conds, numstim, begin, end, num=None, conv=True, 
                  filtsize=10, norm=True, ifr=True):
    stims = np.arange(1, numstim+1)
    stim_list = [[] for x in stims]
    for i, r in enumerate(rasters): # single neuron, many different trials
        for j, s in enumerate(stims):
            inds = np.where(conds[i] == s)
            stimrasts = r[:, inds[1]]
            if ifr:
                window_rates = get_ifr_window(stimrasts, begin, end)
            else:
                window_rates = get_rate_window(stimrasts, begin, end)
            # window_rates is neuronal response in window as function of
            # stimulus presentation number, per stim, per neuron
            if (window_rates == 0).all():
                msg = ('neuron {} has no recorded spikes in window for '
                       'stimulus {} on any trial').format(i, s)
                print msg
            else:
                if norm:
                    window_rates = max_normalize(window_rates)
                if conv:
                    filt = np.ones((filtsize)) / float(filtsize)
                    window_rates = np.convolve(window_rates, filt, mode='valid')
            stim_list[j].append(window_rates)
    return stim_list

def get_neuron_bystim(stimlist, tails=20):
    neurons = []
    for i, neuron in enumerate(stimlist[0]):
        neurons.append([])
        for j, stim in enumerate(stimlist):
            nov = stimlist[j][i][:tails].mean()
            nov_std = stimlist[j][i][:tails].std()
            fam = stimlist[j][i][-tails:].mean()
            fam_std = stimlist[j][i][-tails:].std()
            neurons[i].append((nov, fam, nov_std, fam_std))
    return neurons

def plot_neuron_bystim(neuron):
    fig = plt.figure()
    for x in neuron:
        diffplot = fig.add_subplot(1, 1, 1)
        diffplot.plot(x[:2])
        diffplot.errorbar((0, 1), x[:2], yerr=x[2:])
        diffplot.set_xlim(-.1, 1.1)
    plt.show()

def raster_to_ifr_inverse(rast, binsize=1):
    spike_locs = np.where(rast > 0)[0]
    frtc = np.zeros(rast.shape)
    ifrs = (1000. / binsize) / np.diff(spike_locs)
    for i, fr in enumerate(ifrs):
        frtc[spike_locs[i]:spike_locs[i+1]] = fr
    return frtc

def raster_to_ifr(rast, datsize=1, binsize=20):
    filt = np.ones((binsize)) / float(binsize)
    fr = np.convolve(rast, filt, mode='same')
    correct =  binsize / np.arange(np.ceil(binsize / 2.), binsize)
    fr[:correct.size] = fr[:correct.size] * correct
    fr[-correct.size:] = fr[-correct.size:] * correct[::-1]
    return fr * 1000 / datsize
