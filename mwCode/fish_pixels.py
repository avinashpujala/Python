import sys
sys.path.append(r'C:\Users\woottenm\Documents\Code\zebrafish-analysis')

import numpy as np
from skimage import img_as_int
from skimage.filters import gaussian as gaussian_blur
from skimage.measure import label, regionprops
from snippets.splines import fit_b_spline_to_curve
from snippets.assorted import map_np, zscore, sparse_index, calcDist
from snippets.assorted import find_closest_points, gaussblur
from snippets.threshold import get_multi_threshold


def find_fish_pixels_for_midline_tracking(I, headDiam=1, pxlLen=0.05):
    """
    Given an image stack and a few optional parameters, returns binary images
    with fish pixels set to 1. Assumes that the fish head position is in the
    middle of the image for each image.

    Parameters
    ----------
    I : 3D ndarray (time × rows × columns)
        Image stack in which to find fish pixels
    headDiam : int (optional, default = 1)
        Approximate diameter of the head in millimeters. This determines the
        size of the gaussian kernel used for smoothing, and later for
        thickening connections between fragmented blobs and filling holes in
        fish.
    pxlLen : float (optional, default = 0.05)
        Size of a single pixel in millimeters.

    Returns
    -------
    I_fish : 3D ndarray[bool] (time × rows × columns)
        Image stack, with all fish pixels set to True.

    Author
    -------
    Avinash Pujala (Koyama lab, JRC) in 2018

    """

    headDiam_pxl = round(headDiam / pxlLen)
    kernel_size = np.ceil(headDiam_pxl / 2)
    I_smooth = gaussblur(I, kernel_size)
    multi_thr_out = map_np(lambda img: get_multi_threshold(img, 3), I_smooth)
    thr = map_np(lambda tup: tup[0], multi_thr_out)
    img_quant = map_np(lambda tup: tup[1], multi_thr_out)
    print('thr shape: {}'.format(thr.shape))
    print('img_quant shape: {}'.format(img_quant.shape))
    I_fish = np.array(list(map(lambda a, b: a >= b, I, thr[:, 1])))
    fp = (np.ceil(I.shape[1] / 2), np.ceil(I.shape[2]) / 2)
    imgInds = range(I.shape[0])
    nFishPxls = np.zeros(I.shape[0])
    dists = []
    for jj in imgInds:
        rp = regionprops(label(img_as_int(I_fish[jj])))
        rpDelInds = np.zeros(len(rp), dtype=bool)
        rp = list(filter(lambda p: min(calcDist(p.coords, fp)) < 3, rp))
        # from IPython.core.debugger import Tracer; Tracer()()
        foo = np.zeros_like(I_fish[0])
        if len(rp) > 0:
            for [r, c] in rp[0].coords:
                foo[r, c] = True
            nFishPxls[jj] = rp[0].area
        else:
            nFishPxls[jj] = float('nan')  # could this just be zero?
            print("No fish pixels detected for image {}".format(jj))
        I_fish[jj] = foo
    return I_fish
    # Iterate

    # How this section works:

    # This tries to detect really unusual numbers of fish pixels.
    # We expect that if this varies, it should vary smoothly.
    # We take the differences between this and a smoothed version.
    # If these differences are big, we would see that something is wrong.
    # We determine this by throwing out the top 5% or so (2σ) worst values.

    blah = nFishPxls[~np.isnan(nFishPxls)]
    if len(blah) >= 50:
        blah_smooth = fit_b_spline_to_curve(blah, order=4, smoothness=10)[0]
        eta = zscore(np.abs(blah_smooth-blah))
        blah = blah[~(eta >= 2)]
    # Matlab's std does sample standard deviation by default
    mu = blah.mean() + 1.0 * blah.std(ddof=1)

    if not np.isnan(nFishPxls).all():
        sigma = nFishPxls.std(ddof=1)
        # This is actually 2 standard deviations; I am not sure why 1σ is
        # called "mu"
        overInds = (nFishPxls > (mu + sigma)).nonzero()[0]
        if len(overInds) == 0:
            print('No excess pixel frames!')
        else:
            print(f'Correcting {len(overInds)} with excess fish pxls...')
        itermax = 100
        for jj in overInds:
            thr_new = thr[jj, 1]
            wt = 0.5
            count = 0
            foo = I_smooth[jj]
            T = []
            N = []
            n_prev = np.abs(nFishPxls[jj] - mu)
            shrinking = True
            while (count < itermax) and (np.abs(nFishPxls[jj] - mu) > sigma):
                count += 1
                T.append(thr_new)
                N.append(nFishPxls[jj])
                thr_new = wt * thr[jj, 0] + (1 - wt) * thr[jj, 2]
                blah = (foo >= thr_new)
                nFishPxls[jj] = blah.sum()
                n_now = np.abs(nFishPxls[jj] - mu)
                shrinking = (n_now < n_prev)
                if shrinking:
                    if (nFishPxls[jj] - mu > 0):
                        wt *= 1.05
                    elif (nFishPxls[jj] - mu < 0):
                        wt *= 0.95
                foo = (foo >= thr_new).astype(int)
                # would fill holes with scipy.ndimage.morphology
                # function binary_fill_holes, but commented out
                rp = regionprops(label(img_as_int(foo)))
                if len(rp) > 1:
                    line_inds = set()
                    cp = map_np(lambda x: x.centroid, rp)
                # Connect fragmented fish blobs with straight lines
                for bb in range(len(rp)):
                    blob1 = rp[bb]
                    d = calcDist(cp, blob1.centroid)
                    d[d == 0] = np.inf
                    minInd = d.argmin()
                    blob2 = rp[minInd]
                    closestPts = find_closest_points(blob1.coords, blob2.coords)
                    for points in closestPts[0]:
                        line_x = np.linspace(points[0, 0], points[0, 1], 20)
                        line_y = np.linspace(points[1, 0], points[1, 1], 20)
                        line_inds.add((round(line_y), round(line_x)))
            # Smooth images a bit, especially to thicken connections between
            # formerly fragmented blobs
            blah = foo
            blah[np.array(list(line_inds))] = 1
            I_fish[jj] = blah

        # Line 167 in the Matlab
        underInds = (nFishPxls < (mu - sigma)).nonzero()[0]
        if len(underInds) == 0:
            print('No deficient pixel inds!')
        else:
            print(f'Correcting {len(underInds)} with too few fish pxls...')
        for jj in underInds:
            thr_new = thr[jj, 1]
            wt = 0.5
            count = 0
            foo = I_smooth[jj]
            T = []
            N = []
            n_prev = np.abs(nFishPxls[jj] - mu)
            shrinking = True
            frag = True
            while (
                (count < 100) and
                np.abs(nFishPxls[jj] - mu) > sigma and
                shrinking and
                frag
            ):
                count += 1
                T.append(thr_new)
                N.append(nFishPxls[jj])
                thr_new = wt * thr[jj, 1] + (1 - wt) * 0
                blah = (foo >= thr_new)
                nFishPxls[jj] = blah.sum()
                n_now = np.abs(nFishPxls[jj] - mu)
                shrinking = (n_now < n_prev)
                if shrinking:
                    if (nFishPxls[jj] - mu > 0):
                        wt *= 1.05
                    elif (nFishPxls[jj] - mu < sigma):
                        wt *= 0.95
            foo = (foo >= thr_new).astype(int)
            rp = regionprops(label(img_as_int(foo)))
            if len(rp) > 1:
                line_inds = set()
                cp = map_np(lambda x: x.centroid, rp)
                # Connect fragmented fish blobs with straight lines
                for bb in range(len(rp)):
                    blob1 = rp[bb]
                    d = calcDist(cp, blob1.centroid)
                    d[d == 0] = np.inf
                    minInd = d.argmin()
                    blob2 = rp[minInd]
                    closestPts = find_closest_points(blob1.coords, blob2.coords)
                    for points in closestPts[0]:
                        cp1 = np.squeeze(points[:, 0])
                        cp2 = np.squeeze(points[:, 1])
                        d_pts = np.sqrt(((cp1 - cp2) ** 2).sum()) * pxlLen
                        if d_pts <= (2 * headDiam):
                            line_x = np.linspace(points[0, 0], points[0, 1], 20)
                            line_y = np.linspace(points[1, 0], points[1, 1], 20)
                            line_inds.add((round(line_y), round(line_x)))
            # Smooth images a bit, especially to thicken connections between
            # formerly fragmented blobs
            blah = np.copy(foo)
            blah[np.array(list(line_inds))] = 1
            I_fish[jj] = blah

    ker_s = max([np.ceil(headDiam_pxl / 7), 2])
    print('Thickening binary images by smoothing a bit...')
    I_fish = map_np(lambda img: gaussian_blur(img, ker_s), I_fish)
    # Convert this into binary somehow
    return I_fish
