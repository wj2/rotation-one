
import numpy as np
import scipy.io as sio
import scipy.stats as sst
import scipy.interpolate as sip
import matplotlib.pyplot as plt

# reasonable cutoff for inhib: < 10
# reasonable cutoff for excit: > 20

def figure_one(data, locs, lengths, putative_type, cutoff, begin, end):
    data = load_data(data, locs, lengths=lengths, putative_type=putative_type)
    fam, nov = ready_data(data, cutoff, begin, end)
    currs, nov = plot_data(fam, nov)
    plt.show()
    return currs, nov

def load_data(data, locs, region='it', lengths=None, putative_type='both',
              excitatory_thresh=20, inhibitory_thresh=10):
    data = sio.loadmat(data)
    data = data['data']
    locs = sio.loadmat(locs)
    locs = locs['loc'][:,0]
    if putative_type != 'both':
        lens = sio.loadmat(lengths)['peak2trough'][0]
        if putative_type[0] == 'i':
            places = np.where(lens < inhibitory_thresh)
        elif putative_type[0] == 'e':
            places = np.where(lens > excitatory_thresh)
        data = data[places]
        locs = locs[places]

        loc_locs = np.where(locs == region)
    regional_data = data[loc_locs]
    return regional_data

# using 575 and 700 for begin and end for current dataset
def ready_data(neurons, cutoff, begin, end):
    """ 
    receive ixjxk array where i represents different neurons, j different 
    stimuli, and k different measurements in time
    """
    neurons = window_of_interest(neurons, begin, end)
    fam = neurons[:, :cutoff, :]
    nov = neurons[:, cutoff:, :]
    fam = fam.mean(2)
    nov = nov.mean(2)
    # now we have ixnovel_stim and ixfamiliar_stim averaged over the window of
    # interest
    # reduce to same number of stim for last plot
    # fam = fam[:, :7]
    # nov = nov[:, :7]
    return fam.flatten(), nov.flatten()

def window_of_interest(data, t_b, t_e):
    """ stimulus is presented 500ms - 1150ms """
    # data should be three dimensional: neuron, stimulus, timecourse
    return data[:, :, t_b:t_e+1]

def distribution_of_firing(axes, rates, log10=False, plot=True, label=None):
    if log10:
        rates = np.log10(rates)
    if plot:
        bins, edges, _ = axes.hist(rates, bins=30, histtype='step', label=label)
        axes.legend()
    else:
        bins, edges = np.histogram(rates, bins=30)
    bins = bins / float(len(rates))
    return bins, edges

def quantile_frate(ratetrain):
    percs = [sst.percentileofscore(ratetrain, el) / 100. 
             for el in ratetrain]
    return percs

def rate_score(ratetrain, percs=1):
    percentiles = np.arange(0, 101, percs)
    percs = np.array([np.percentile(ratetrain, p) for p in percentiles])
    return percs, percentiles / 100.
    
def quantile_inputcurr(percentiles, mean=0, std=1, sanitize_inf=False,
                       eps=.00001):
    ps = sst.norm(loc=mean, scale=std).ppf(percentiles)
    if sanitize_inf:
        if np.isinf(ps[0]):
            ps[0] = sst.norm(loc=mean, scale=std).ppf(0 + eps)
        if np.isinf(ps[-1]):
            ps[-1] = sst.norm(loc=mean, scale=std).ppf(1 - eps)
    return ps

def sanitize_infinities_monotonic(seq):
    if np.isinf(seq[0]):
        seq[0] = seq[1] * 2
    if np.isinf(seq[-1]):
        seq[-1] = seq[-2] * 2
    return seq

def plot_data(fam, nov, title=None, log=False):
    # top line
    fig = plt.figure(figsize=(10, 6))
    if title is not None:
        fig.suptitle(title)
    hists = fig.add_subplot(2, 4, 1)
    # distribution_of_firing(hists, fam, label='familiar')
    distribution_of_firing(hists, nov, log10=log, label='novel')
    nov = np.sort(nov)
    fam = np.sort(fam)

    percentiles = quantile_frate(nov)
    frate_func = fig.add_subplot(2,4,2)
    frate_func.plot(percentiles, nov)

    currents = quantile_inputcurr(percentiles, sanitize_inf=True)
    curr_func = fig.add_subplot(2,4,3)
    curr_func.plot(percentiles, currents)
    
    transfer_func = fig.add_subplot(2,4,4)
    transfer_func.plot(currents, nov, 'o', markersize=1)

    # bottom line 
    both_hists = fig.add_subplot(2, 4, 5)
    distribution_of_firing(both_hists, nov, label='novel')
    distribution_of_firing(both_hists, fam, label='familiar')
    
    percentiles_fam = quantile_frate(fam)
    both_frate = fig.add_subplot(2,4,6)
    both_frate.plot(percentiles, nov, label='novel')
    both_frate.plot(percentiles_fam, fam, label='familiar')

    both_currents = fig.add_subplot(2, 4, 7)
    both_currents.plot(percentiles, currents, label='novel')
    # interp_func = sip.interp1d(nov, currents)
    order = nov.argsort()
    # explicitly remove inf value from this
    nov_unique, order = np.unique(nov, return_index=True)
    # nov_unique = nov[order]
    currs_unique = sanitize_infinities_monotonic(currents[order])
    bbright = np.max([fam.max(), nov_unique[-1]])
    bbleft = np.min([fam.min(), nov_unique[0]])
    interp_func = sip.InterpolatedUnivariateSpline(nov_unique, currs_unique, 
                                                   k=1, 
                                                   bbox=[bbleft, bbright])
    currents_fam = interp_func(fam)
    both_currents.plot(percentiles_fam, currents_fam, label='familiar')
    input_change = fig.add_subplot(2, 4, 8)
    currfam_score, _ = rate_score(currents_fam)
    currnov_score, _ = rate_score(currents)
    currdiff_score = currfam_score - currnov_score
    nov_score, _ = rate_score(nov)
    # print currfam_score, currnov_score
    input_change.plot(nov_score, currdiff_score, 'o', markersize=1)
    input_change.plot(nov_score, currdiff_score, 'o', markersize=1)
    # delta_curr = np.array(currents_fam_unsorted) - np.array(currents_unsorted)
    # input_change.plot(nov, delta_curr, 'o', markersize=1)

    return currents, nov


def plot_data_percs(fam, nov):
    # top line
    fig = plt.figure()
    hists = fig.add_subplot(2, 4, 1)
    # distribution_of_firing(hists, fam, label='familiar')
    distribution_of_firing(hists, nov, label='novel')
    nov = np.sort(nov)
    fam = np.sort(fam)

    nov_scores, percs = rate_score(nov)
    frate_func = fig.add_subplot(2,4,2)
    frate_func.plot(percs, nov_scores)

    currents = quantile_inputcurr(percs, sanitize_inf=True)
    curr_func = fig.add_subplot(2,4,3)
    curr_func.plot(percs, currents)
    
    transfer_func = fig.add_subplot(2,4,4)
    transfer_func.plot(currents, nov_scores, 'o', markersize=1)

    # bottom line 
    both_hists = fig.add_subplot(2, 4, 5)
    distribution_of_firing(both_hists, nov, label='novel')
    distribution_of_firing(both_hists, fam, label='familiar')
    
    fam_scores, percs = rate_score(fam)
    both_frate = fig.add_subplot(2,4,6)
    both_frate.plot(percs, nov_scores, label='novel')
    both_frate.plot(percs, fam_scores, label='familiar')

    both_currents = fig.add_subplot(2, 4, 7)
    both_currents.plot(percs, currents, label='novel')
    # interp_func = sip.interp1d(nov, currents)
    # order = nov.argsort()
    # explicitly remove inf value from this
    # nov_unique, order = np.unique(nov, return_index=True)
    # nov_unique = nov[order]
    # currs_unique = sanitize_infinities_monotonic(currents[order])
    bbright = np.max([fam_scores.max(), nov_scores[-1]])
    bbleft = np.min([fam_scores.min(), nov_scores[0]])
    interp_func = sip.InterpolatedUnivariateSpline(nov_scores, currents, 
                                                   k=1, 
                                                   bbox=[bbleft, bbright])
    currents_fam = interp_func(fam_scores)
    both_currents.plot(percs, currents_fam, label='familiar')
    input_change = fig.add_subplot(2, 4, 8)
    # delta_curr = np.array(currents_fam_unsorted) - np.array(currents_unsorted)
    # input_change.plot(nov, delta_curr, 'o', markersize=1    

def avg_neuron(neuron, cutoff):
    familiar = neuron[:cutoff].mean(1)
    novel = neuron[cutoff:].mean(1)
    return familiar, novel
           
    
