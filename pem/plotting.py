import pem.io

from matplotlib.pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import matplotlib.gridspec as gridspec

colors = dict(core=sns.xkcd_rgb['tan'],env=sns.xkcd_rgb['light blue'])

def label_prad_yaxis():
    ylabel('Planet Size (Earth-radii)')
    yt = [1,2,3,4,5,6,7,8,9,10,20]
    yticks(yt,yt)
    ylim(0.5,10)

def label_per_xaxis():
    xlabel('Period (days)')
    xt = [1,3,10,30,100]
    xticks(xt,xt)
    xlim(xt[0],xt[-1])

def label_time_xaxis():
    xlabel('Time (years)')
    xlim(1e4,1e9)
    xt = [1e4,1e5,1e6,1e7,1e8,1e9]
    xticks(xt,xt)

def _provision_figure(plot_perprad=True, plot_radii=False, plot_timeline=False):
    """
    See per_prad_movie
    """
    sns.set_context('talk',font_scale=0.9)
    fig = figure(figsize=(10,6))
    gs1 = gridspec.GridSpec(100, 100)
    axL = []
    if plot_perprad:
        ax1 = plt.subplot(gs1[0:80, 0:60])
        axL.append(ax1)
        loglog()
        label_prad_yaxis()
        label_per_xaxis() 

    if plot_timeline:
        ax2 = plt.subplot(gs1[95:100, 0:60])
        axL.append(ax2)

    if plot_radii:
        ax3 = plt.subplot(gs1[0:80, 70:100])
        axL.append(ax3)
        semilogy()
        label_prad_yaxis()
        sxt = [0,100,200,300]
        xt = np.array(sxt)*1e6
        xticks(xt,sxt)
        xlim(0,3e8)
        xlabel('Time (Myr)')
        #        label_time_xaxis()

    return fig, axL 
    
def per_prad_movie(mode, giffn, plot_population=False, plot_iplanet=-1, 
                   plot_perprad=True, plot_radii=False, plot_timeline=False,
                   step=200):
    """
    Args:

        plot_iplanet (int): index of single planet to plot, if -1 then
            don't plot at all

        plot_timeline (bool): make a scale bar with the timeline
        plot_radii (bool): plot radii of core and envelope

    """

    df = pem.io.load_table(mode,cache=1)
    fig, axL = _provision_figure(
        plot_perprad=plot_perprad, plot_radii=plot_radii, 
        plot_timeline=plot_timeline
    )

    df = df[df.time.between(1e4,1e9)]

    # Set initial figures
    itime_min = df.iloc[0]['itime']
    itime_max = df.iloc[-1]['itime']

    fig = gcf()
    if plot_population:
        sca(axL[0])
        cut = df[df.itime==itime_min]
        x = cut.per
        y = cut.radius_total
        pop, = plot(x, y, '.',)

    if plot_iplanet >= 0:
        sca(axL[0])
        row = df[(df.itime==itime_min) & (df.iplanet==plot_iplanet)]
        x,y = row.per,row.radius_total
        fac = 20
        ms_core = fac*row.radius_core
        ms_total = fac*row.radius_total
        core, = plot(x, y, '.', ms=ms_core, zorder=2.1, color=colors['core'])
        env, = plot(x, y, '.', ms=ms_total, zorder=2, color=colors['env'])

    if plot_radii:
        sca(axL[1])
        cut = df[df.iplanet==plot_iplanet]
        plot(cut.time,cut.radius_core, color=colors['core'], zorder=2.1)
        plot(cut.time,cut.radius_total, color=colors['env'], zorder=2.0)

        x,y = row.per,row.radius_total
        radii_point, = plot(x, y, 'x',ms=10,zorder=3,mew=2)

    def animate(i):
        print i
        if plot_population:
            cut = df[df.itime==i]
            y = cut.radius_total
            pop.set_ydata(y)  # update the data
     
        if plot_iplanet>=0:
            row = df[(df.itime==i) & (df.iplanet==plot_iplanet)]
            core.set_ydata(row.radius_total)
            env.set_ydata(row.radius_total)

            ms_core = fac*row.radius_core
            ms_total = fac*row.radius_total
            core.set_markersize(ms_core)
            env.set_markersize(ms_total)

        if plot_radii:
            row = df[(df.itime==i) & (df.iplanet==plot_iplanet)]
            radii_point.set_xdata(row.time)
            radii_point.set_ydata(row.radius_total)

    '''
    # Init only required for blitting to give a clean slate.
    def init():
        pop.set_ydata(np.ma.array(x, mask=True))
        return pop,

    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, 1000,50), init_func=init,
        interval=25, blit=True
    '''

    ani = animation.FuncAnimation(fig, animate, np.arange(itime_min, itime_max,step))
    ani.save(giffn, writer='imagemagick', fps=30)


def _plot_iplanet(row):
    fac = 20
    colors = dict(core=sns.xkcd_rgb['tan'],env=sns.xkcd_rgb['pale pink'])
    x,y = row.per,row.radius_total
    ms_core = fac*row.radius_core
    ms_total = fac*row.radius_total
    plot(x, y, '.', ms=ms_core, zorder=2.1, color=colors['core'])
    plot(x, y, '.', ms=ms_total, zorder=2, color=colors['env'])


from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline


class Plotter(object):
    def __init__(self, mode):
        df = pem.io.load_table(mode,cache=1)
        df['logtime'] = np.log10(df.time)

        # Define the regular grid for interpolation
        x0 = np.array(df.iplanet.drop_duplicates()) # (nplanets)
        x1 = np.array(df.logtime.drop_duplicates()) # (timesteps)
        nplanet = len(x0)
        ntime = len(x1)

        points  = (x0, x1) 
        cols = 'radius_total'.split()
        values = df[cols] # (nplanet, 1)
        values = array(df[cols]).reshape(nplanet,ntime)
        interp = RegularGridInterpolator(points, values, bounds_error=False, fill_value=None)

        self.df = df
        self.interp = interp
        self.x0 = x0
        self.nplanet = nplanet
        
    def get_snapshot(self, time):
        """
        Return population at a given time
        """

        logtime = np.log10(time)

        x0i = self.x0
        x1i = ones(self.nplanet)*logtime
        pointsi = np.vstack([x0i,x1i]).T
        df = self.df[self.df.itime==self.df.itime.min()].copy()['iplanet per radius_core'.split()]
        df['radius_total'] = self.interp(pointsi)
        df['time'] = time
        return df

    def plot_frame(self, time, plot_population=False,
                   plot_iplanet=False, iplanet=0, plot_perprad=True,
                   plot_radii=False, plot_timeline=False):
        """
        Args:
        
            plot_iplanet (int): index of single planet to plot, if -1 then
            don't plot at all

            plot_timeline (bool): make a scale bar with the timeline
            plot_radii (bool): plot radii of core and envelope
        
        """

        fig, axL = _provision_figure(
            plot_perprad=plot_perprad, plot_radii=plot_radii, 
            plot_timeline=plot_timeline
        )


        plnt_all = self.df[self.df.iplanet==iplanet]
        pop = self.get_snapshot(time)
        plnt = pop[pop.iplanet==iplanet] # values for iplanet for given frame
       
        if plot_population:
            sca(axL[0])
            x = pop.per
            y = pop.radius_total
            plot(x, y, '.',zorder=1,ms=5)

        if plot_iplanet:
            sca(axL[0])
            x, y = plnt.per,plnt.radius_total
            fac = 20
            ms_core = fac*plnt.radius_core
            ms_total = fac*y
            plot(x, y, '.', ms=ms_core, zorder=2.1, color=colors['core'])
            plot(x, y, '.', ms=ms_total, zorder=2, color=colors['env'])

        if plot_radii:
            sca(axL[1])
            x = plnt_all.time
            y = plnt_all.radius_core
            plot(x,y, color=colors['core'], zorder=2.1)

            y = plnt_all.radius_total
            plot(x,y, color=colors['env'], zorder=2.0)

            x,y = plnt.time, plnt.radius_total
            plot(x, y, 'x',ms=10,zorder=3,mew=2)

            print plnt.per,plnt.radius_total
    
    def save_frames(self, times, prefix, **kwargs):
        """
        """
        i = 0
        for time in times:
            self.plot_frame(time, **kwargs)
            pngfn = '{}{:04d}.png'.format(prefix,i)
            gcf().savefig(pngfn)
            i+=1

