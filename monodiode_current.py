import numpy as np
from scipy import stats
import warnings

def meanclip(indata, clipsig=3.0, maxiter=5, converge_num=0.02, verbose=0):
   """
   Computes an iteratively sigma-clipped mean on a
   data set. Clipping is done about median, but mean
   is returned.

   .. note:: MYMEANCLIP routine from ACS library.

   :History:
       * 21/10/1998 Written by RSH, RITSS
       * 20/01/1999 Added SUBS, fixed misplaced paren on float call, improved doc. RSH
       * 24/11/2009 Converted to Python. PLL.

   Examples
   --------
   >>> mean, sigma = meanclip(indata)

   Parameters
   ----------
   indata: array_like
       Input data.

   clipsig: float
       Number of sigma at which to clip.

   maxiter: int
       Ceiling on number of clipping iterations.

   converge_num: float
       If the proportion of rejected pixels is less than
       this fraction, the iterations stop.

   verbose: {0, 1}
       Print messages to screen?

   Returns
   -------
   mean: float
       N-sigma clipped mean.

   sigma: float
       Standard deviation of remaining pixels.

   """
   # Flatten array
   skpix = indata.reshape( indata.size, )

   ct = indata.size
   iter = 0; c1 = 1.0 ; c2 = 0.0

   while (c1 >= c2) and (iter < maxiter):
       lastct = ct
       medval = np.median(skpix)
       sig = np.std(skpix)
       wsm = np.where( abs(skpix-medval) < clipsig*sig )
       ct = len(wsm[0])
       if ct > 0:
           skpix = skpix[wsm]

       c1 = abs(ct - lastct)
       c2 = converge_num * lastct
       iter += 1
   # End of while loop

   mean  = np.mean( skpix )
   #sigma = robust_sigma( skpix )
   sigma = skpix.std()

   if verbose:
       prf = 'MEANCLIP:'
       print '%s %.1f-sigma clipped mean' % (prf, clipsig)
       print '%s Mean computed in %i iterations' % (prf, iter)
       print '%s Mean = %.6f, sigma = %.6f' % (prf, mean, sigma)

   return mean, sigma

def monodiode_current(inputdat):
    data = inputdat[17].data
    x_t, y_pA = data.field('AMP0_MEAS_TIMES'), data.field('AMP0_A_CURRENT')

#    print "Raw data:"
#    for i in range(len(x_t)):
#        print str(i) + '\t' + str(x_t[i]) + '\t' + str(y_pA[i])
#    exit()


    if (np.all(x_t == np.zeros(len(x_t)))):
        warnings.warn("RuntimeWarning: AMP0_MEAS_TIMES binary table has no measurments!")
        return (0, 0)


    # Test for const value 1st...
    # m ~= 0 we assume a line, meaning bias image (or no light) or const value
    # just return clipped mean of the whole time series
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_t, y_pA)
    if (np.abs(np.round(slope)) == 0.0): #and np.abs(np.round(intercept)) == 0.0):
        return (meanclip(y_pA))


    # Not a const value? Try to find the real signal...
    # There must be a step in the time series, so find the critical points
    i = 0;
    cirt_pnts = [];
    downflg = 0; upflg = 0;

    # Normalize the data [0, 1]
    norm = y_pA / np.max(np.abs(y_pA));

    while (i < len(norm) - 2):
        # Check thresholds (-0.8 for now)
        if (norm[i] <= -0.8 and downflg == 0):
            # Look for dy/dt going DOWN within a +/- 2 point window
            if (np.sum(np.diff(y_pA[i-2:i+2])) < 0.0):
                downflg = 1;
                print "v Found transition at t=", x_t[i],  y_pA[i], i
                cirt_pnts.append(i);
        elif (norm[i] >= -0.8 and upflg == 0 and downflg == 1):
            #  Look for dy/dt going UP within a +/- 2 point window
            if (np.sum(np.diff(y_pA[i-2:i+2])) > 0.0):
                upflg = 1;
                print "^ Found transition at t=", x_t[i],  y_pA[i], i
                cirt_pnts.append(i);
                break;
        i += 1;

#     print "Merlin:"
#     for value in cirt_pnts:
#         print value

    # If we found any critical points (max = 2)
    # report the mean within the level, if no upflag set, assume t1 hits
    # the end of the time series.
    if (len(cirt_pnts) > 0):
        x1 = cirt_pnts[0];

        # If we never found the signal returning to zero, assume it continues
        # to the end of the time series...
        if (upflg == 0):
            x2 = len(y_pA) - 1;
        else:
            x2 = cirt_pnts[len(cirt_pnts)-1];

        return (meanclip(y_pA[x1:x2]), (x1, x2))
    else: # Give up, tell the user we failed and return a blind mean, std value
        warnings.warn("RuntimeWarning: no critical points found, possible inaccurate return value in monodiode_current")
        return (meanclip(y_pA))

