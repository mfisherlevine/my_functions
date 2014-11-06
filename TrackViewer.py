import pylab as pl
import numpy
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from os.path import join
import matplotlib.pyplot as plt

def _makeVectorRedundant(xs, red):
    t = numpy.zeros(red * xs.shape[0])
    for i in xrange(red):
        t[i::red] = xs
    return t

def _prepLegoData(xlims, ylims, zvals):
    """Prepare 3D histogram data to be used with matplotlib's Axes3D/Axes3DI.

    usage example:

    >>> nx, ny = 3, 5
    >>> X, Y, Z = _prepLegoData(numpy.arange(nx), numpy.arange(ny),
    ... numpy.random.rand(nx, ny))
    >>> fig = pl.figure()
    >>> ax = matplotlib.axes3d.Axes3DI(fig)
    >>> ax.plot_surface(X, Y, Z, rstride=2, cstride=2)
    >>> pl.show()

    @param xlims: N+1 array with the bin limits in x direction
    @param ylims: M+1 array with the bin limits in y direction
    @param zvals: a 2D array with shape (N, M) with the bin entries,
        example::
        --> y-index (axis 1)
        |       z_0_0  z_0_1  ...
        |       z_1_0  z_1_1  ...
        V        ...
        x-index (axis 0)
    @returns: X, Y, Z 2D-arrays for Axes3D plotting methods
    """
    
    assert xlims.shape[0] - 1 == zvals.shape[0]
    assert ylims.shape[0] - 1 == zvals.shape[1]

    # use a higher redundancy for surface_plot
    # must be a multiple of 2!
    red = 4

    X, Y = pl.meshgrid(_makeVectorRedundant(xlims, red),
                                    _makeVectorRedundant(ylims, red))
    #X, Y = matplotlib.mlab.meshgrid(_makeVectorRedundant(xlims, 2),
                         #_makeVectorRedundant(ylims, 2))
    Z = numpy.zeros(X.shape)

    # enumerate the columns of th zvals
    for yi, zvec in enumerate(zvals):
        #print _makeVectorRedundant(zvec, red)
        #print Z[2*xi + 1, 1:-1]
        #Z[2*xi + 1, 1:-1] = Z[2*xi + 2, 1:-1] = _makeVectorRedundant(zvec, 2)
        t = _makeVectorRedundant(zvec, red)
        #print 't', t.shape
        for off in xrange(1, red+1):
            #print red*xi, Z[red*xi + off, red/2:-red/2].shape
            # Z[red*xi + off, red/2:-red/2] = t
            Z[red/2:-red/2, red*yi + off] = t
    return X, Y, Z


def TrackToFile_ROOT(data, save_path, log_z = False, plot_opt = '', force_aspect = True, legend_text = '', fitline = None):
    from ROOT import TH2F, TCanvas
    if plot_opt == '': plot_opt = 'lego2 0'
    if plot_opt != 'lego' and plot_opt != 'box' and plot_opt != 'colz' and plot_opt != 'lego2' and plot_opt != 'lego2 0': plot_opt = 'lego2 0'
    
    if force_aspect:
        nbinsx = xmax = max(data.shape[0], data.shape[1])
        nbinsy = ymax = max(data.shape[0], data.shape[1])
    else:
        nbinsx = xmax = data.shape[0]
        nbinsy = ymax = data.shape[1]
    
    xlow = 0
    ylow = 0
    image_hist = TH2F('', '',nbinsx,xlow,xmax,nbinsy, ylow, ymax)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            value = data[x,y]
            if value != 0:
                image_hist.Fill(float(x),float(y),float(value))
        
    c4 = TCanvas( 'canvas', 'canvas', 0, 0, 1000, 1000 ) #create canvas
    image_hist.Draw(plot_opt)

    if legend_text != '':
        from ROOT import TPaveText
        textbox = TPaveText(0.80,0.90,1.0,1.0,"NDC")
        for line in legend_text:
            textbox.AddText(line)
        textbox.Draw("same")
        
    if fitline != None:
        from ROOT import TF2
        a = fitline.a
        b = fitline.b
        
#        formula = "y - (" + str(a) + "*x) - "+ str(b) + " - 5"
        formula = "(" + str(a) + "*x) - "+ str(b)
        print formula
        
#        formula = "sin(x)*sin(y)/(x*y)"
        
        plane = TF2("f2",formula,0,20,0,20)
        plane.SetContour(100)
        plane.Draw("same")
        

#    from root_functions import Browse
#    image_hist.SetDrawOption(plot_opt)
#    Browse(image_hist)
        
    if log_z: c4.SetLogz()
    image_hist.SetStats(False)
    c4.SaveAs(save_path)
    
    
def TrackToFile_ROOT_2D(data, save_path, log_z = False, plot_opt = '', force_aspect = True, legend_text = '', fitline = None):
    from ROOT import TH2F, TCanvas
    if plot_opt == '': plot_opt = 'colz'
    if plot_opt != 'box' and plot_opt != 'colz': plot_opt = 'colz'
    
    if force_aspect:
        nbinsx = xmax = max(data.shape[0], data.shape[1])
        nbinsy = ymax = max(data.shape[0], data.shape[1])
    else:
        nbinsx = xmax = data.shape[0]
        nbinsy = ymax = data.shape[1]
    
    xlow = 0
    ylow = 0
    image_hist = TH2F('', '',nbinsx,xlow,xmax,nbinsy, ylow, ymax)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            value = data[x,y]
            if value != 0:
                image_hist.Fill(float(x),float(y),float(value))
        
    c4 = TCanvas( 'canvas', 'canvas', 0, 0, 1000, 1000 ) #create canvas
    image_hist.Draw(plot_opt)

    if legend_text != '':
        from ROOT import TPaveText
        textbox = TPaveText(0.80,0.90,1.0,1.0,"NDC")
        for line in legend_text:
            textbox.AddText(line)
        textbox.Draw("same")
        
    if fitline != None:
        from ROOT import TF1
        a = fitline.a
        b = fitline.b
        formula = str(a) + '*x + ' + str(b)
        plane = TF1("f2",formula,0,xmax)
        plane.SetNpx(1000)
        plane.Draw("same")
        
    if log_z: c4.SetLogz()
    image_hist.SetStats(False)
    c4.SaveAs(save_path)
    
    
def TrackToFile_ROOT_2D_3D(data, save_path, log_z = False, plot_opt = '', force_aspect = True, legend_text = '', fitline = None):
    from ROOT import TH2F, TCanvas
    if plot_opt == '': plot_opt = 'colz'
    if plot_opt != 'box' and plot_opt != 'colz': plot_opt = 'colz'
    
    if force_aspect:
        nbinsx = xmax = max(data.shape[0], data.shape[1])
        nbinsy = ymax = max(data.shape[0], data.shape[1])
    else:
        nbinsx = xmax = data.shape[0]
        nbinsy = ymax = data.shape[1]
    
    xlow = 0
    ylow = 0
    
    c4 = TCanvas( 'canvas', 'canvas', 1600,800) #create canvas
    c4.Divide(2,1,0.002,0.00001)
    c4.cd(2)
    
    image_hist = TH2F('', '',nbinsx,xlow,xmax,nbinsy, ylow, ymax)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            value = data[x,y]
            if value != 0:
                image_hist.Fill(float(x),float(y),float(value))
        
    image_hist.Draw(plot_opt)
    if fitline != None:
        from ROOT import TF1
        a = fitline.a
        b = fitline.b
        formula = str(a) + '*x + ' + str(b)
        plane = TF1("f2",formula,0,xmax)
        plane.SetNpx(100)
        plane.Draw("same")
        
        
    c4.cd(1)
    image_hist.Draw("lego20")

    if legend_text != '':
        from ROOT import TPaveText
        textbox = TPaveText(0.0,1,0.2,0.9,"NDC")
        for line in legend_text:
            textbox.AddText(line)
        textbox.SetFillColor(0)
        textbox.Draw("same")

        
    if log_z: c4.SetLogz()
    image_hist.SetStats(False)
    c4.SaveAs(save_path)
        

def TrackToFile_MPL(data, save_path, track_name = ''):
    nx = data.shape[0]
    ny = data.shape[1]
    
    X, Y, Z = _prepLegoData(numpy.arange(nx + 1), numpy.arange(ny + 1), data)
    
    fig = plt.figure()
    ax3 = axes3d.Axes3D(fig)
    ax3.plot_surface(X, Y, Z, rstride=2, cstride=2, edgecolors='w', cmap=cm.jet)

    ax3.set_title(track_name)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('ADU')
    
    ax3.auto_scale_xyz([0, max([data.shape[0], data.shape[1]])], [0, max([data.shape[0], data.shape[1]])], [0, numpy.amax(Z)])
    
    fig.savefig(save_path)
    del fig


def ViewTrack(data, save_path = '', track_name = ''):
    nx = data.shape[0]
    ny = data.shape[1]
    
    X, Y, Z = _prepLegoData(numpy.arange(nx + 1), numpy.arange(ny + 1), data)
    
    fig = plt.figure()
#    ax3 = axes3d.Axes3D(fig)
    ax3 = fig.gca(projection = '3d', aspect = 'equal')

    
    ax3.plot_surface(X, Y, Z, rstride=2, cstride=2, edgecolors='w', cmap=cm.jet)
    ax3.set_title(track_name)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('ADU')
    
    ax3.axis('equal')
    
    

    
    
    ax3.auto_scale_xyz([0, max([data.shape[0], data.shape[1]])], [0, max([data.shape[0], data.shape[1]])], [0, numpy.amax(Z)])
    ax3.auto_scale_xyz([0, max([data.shape[0], data.shape[1]])], [0, max([data.shape[0], data.shape[1]])], [0, numpy.amax(Z)])
    ax3.auto_scale_xyz([0, max([data.shape[0], data.shape[1]])], [0, max([data.shape[0], data.shape[1]])], [0, numpy.amax(Z)])
#    ax3.pbaspect = [1,1,1]   

    #    ax3.view_init(elev=10., azim=20)
#    ax3.dist=20

#    ax3.set_xlim(0, data.shape[0])
#    ax3.set_ylim(0, data.shape[1])
#    ax3.set_zscale('log')


    
    if save_path != '':
        fig.savefig(join(save_path, track_name + '.jpg'))

    plt.show()
    del fig