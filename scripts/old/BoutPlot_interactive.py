import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mc
from matplotlib import cm

from boutdata.data import BoutData
from Readers.BOUT_Reader import BOUT_data
from boututils.datafile import DataFile
from matplotlib.widgets import Button, Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fftn as FFTN
from scipy.fft import fftfreq as FREQ

class InteractivePlot:

    def __init__(self,function,x,plot_vars,slider_vars,x_label,var_labels,slider_labels):

    # load data

        self.function = function
        self.x = x
        self.plot_vars = plot_vars
        self.slider_vars = slider_vars
        #print('plot_vars = ',self.plot_vars)
        #print('slider_vars = ',self.slider_vars)
        self.sliders = []

        self.x_label = x_label
        self.var_labels = var_labels
        self.slider_labels = slider_labels

        if isinstance(plot_vars,list):
            self.n_plots = len(plot_vars)
        else:
            self.n_plots = 1

        print('n_plots = ',self.n_plots)
        self.plot()

    def execute_function(self,plot_vars,sliders):
        out = self.function(plot_vars,sliders)
        return out

    def update(self,val):
        #Update plotted data when slider is changed
        for i in range(0,self.n_plots):
            y_data = self.execute_function(self.plot_vars[i],self.sliders) #self.sliders[0].val)
            self.int_plot[i][0].set_ydata(y_data)
            self.axs[i].set_ylim(np.min(y_data),np.max(y_data))
        self.fig.canvas.draw_idle()

    def plot(self):
        
        self.fig,self.axs = plt.subplots(nrows=self.n_plots, ncols=1)

        for i, svar in enumerate(self.slider_vars):
            slider_loc = [0,0.05,0.1]
            slider_ax = self.fig.add_axes([0.25, slider_loc[i], 0.65, 0.03])
            self.sliders.append(Slider(
                                ax=slider_ax,
                                label=self.slider_labels[i],
                                valmin=np.min(svar),
                                valmax=np.max(svar),
                                valinit=0,
                                valstep=svar))
            
        self.slider_vals = [s.val for s in self.sliders]

        self.int_plot = []
        for i in range(0,self.n_plots):
            print('generating plot ',i+1)
            self.int_plot.append(self.axs[i].plot(self.x,self.execute_function(self.plot_vars[i],self.sliders))) #self.sliders[0].val))
            self.axs[i].set_ylabel(self.var_labels[i])
            self.axs[i].set_xlim(0,None)

        for slider in self.sliders:
            slider.on_changed(self.update)

        self.axs[-1].set_xlabel(self.x_label)

        plt.subplots_adjust(bottom=0.2)

        plt.show()

class InteractivePlot2D:

    def __init__(self,data_dir,bgrid_file_loc,numpy_dir):

         # Collect grid quantities
        d = BoutData(path=data_dir)
        print(d["outputs"]["t_array"])
        self.nt = d["outputs"]["t_array"].shape[0] #int(d["outputs"]["Ni"].shape[0]) d["outputs"]["t"].shape[0]
        self.nx = int(d["outputs"]["Ni"].shape[1])
        self.ny = int(d["outputs"]["Ni"].shape[2])
        self.nz = int(d["outputs"]["Ni"].shape[3])
        print("nt nx ny nz = ",self.nt,self.nx,self.ny,self.nz)

        with DataFile(bgrid_file_loc) as g:
            self.rxy = g["Rxy"]
            self.zxy = g["Zxy"]
            #Bpxy = g["Bpxy"]
            #Btxy = g["Btxy"]

        #bout=BOUT_data(bgrid_file_loc)
        #self.rxy = bout.Rxy
        #self.zxy = bout.Zxy
        #self.nx,self.ny=bout.Rxy.shape # here (x,y)=(radial,poloidal)
        self.x = range(0,self.nx+1)
        self.y = range(0,self.ny+1)
        self.t = 0

        self.numpy_dir = numpy_dir

        self.cmap = 'cividis'
        self.norm = mc.TwoSlopeNorm(0)
        self.norm_type = 'linear'
        self.plot()

    def load(self,input_var):
        out = np.load(self.numpy_dir / (input_var+".npy"))
        print(out.shape)
        return out

    def execute_function(self):
        #print("ndim = ",self.ndim)
        if self.ndim == 4:
            out = self.var[self.t,:,:,self.z]
        if self.ndim == 3:
            out = self.var[self.t,:,:]
        if self.ndim == 2:
            out = self.var[:,:]
        return out
    
    def update(self):
        #Execute function
        
        values = self.execute_function()
        val_min = np.min(values)
        #print("val_min = ",val_min)
        val_max = np.max(values)
        #print("val_max = ",val_max)
        if val_min >= 0:
            self.cmap = 'Blues'
            if self.norm_type == 'linear':
                self.norm = mc.Normalize(val_min,val_max)
            if self.norm_type == 'log':
                self.norm = mc.LogNorm(val_min,val_max)
        elif val_max <= 0:
            self.cmap = 'Blues_r'
            if self.norm_type == 'linear':
                self.norm = mc.Normalize(val_min,val_max)
            if self.norm_type == 'log':
                self.norm = mc.SymLogNorm(0.0001*val_min,1,val_min,val_max)
        else:
            self.cmap = 'cividis'
            if self.norm_type == 'linear':
                self.norm = mc.TwoSlopeNorm(0,val_min,val_max)
            if self.norm_type == 'log':
                self.norm = mc.SymLogNorm(0.0001*val_max,1,val_min,val_max)
        self.pcs_ax.clear()
        self.mtx_ax.clear()

        pcs_ax_plot = self.pcs_ax.pcolor(self.rxy, self.zxy, values, cmap=self.cmap, norm=self.norm)
        mtx_ax_plot = self.mtx_ax.pcolormesh(self.y, self.x, values, cmap=self.cmap, norm=self.norm, shading='auto')
        if int(self.ndim) == 4:
            self.mtx_ax.set_title("t = %i   z = %i" % (self.t+1, self.z+1))
        if int(self.ndim) == 3:
            self.mtx_ax.set_title(str(self.t))
        if int(self.ndim) == 2:
            self.mtx_ax.set_title("Plasma Background")
        self.cbar = plt.colorbar(mtx_ax_plot,cax=self.cbar_ax)
        self.fig.canvas.draw_idle()

    def change_plot_var(self,input_var):
        self.var = self.load(input_var)
        self.ndim = len(self.var.shape)
        self.pcs_ax.clear()
        self.mtx_ax.clear()
        self.update()

    def on_press(self,event):

        if event.key == 'up':
            #increase z
            self.z += 1
            if self.z > self.nz-1:
                self.z = self.nz-1
            self.update()

        if event.key == 'down':
            #decrease z
            self.z -= 1
            if self.z < 0:
                self.z = 0
            self.update()

        if event.key == 'left':
            #decrease time
            self.t -= 1
            if self.t < 0:
                self.t = 0
            self.update()

        if event.key == 'right':
            #increase time
            self.t += 1
            if self.t > self.nt-1:
                self.t = self.nt-1
            self.update()

        if event.key == 'control':
            if self.norm_type == 'linear':
                self.norm_type == 'log'
            if self.norm_type == 'log':
                self.norm_type == 'linear'
            self.update()

        if event.key == 'pagedown':
            self.t = 0
            self.update()

        if event.key == 'pageup':
            self.t = self.nt-1
            self.update()

    def plot(self):

        self.t = 0
        self.z = 0

        #Create Fig
        
        fig,(pcs_ax,mtx_ax,cbar_ax) = plt.subplots(1,3)

        self.fig = fig
        self.pcs_ax = pcs_ax
        self.mtx_ax = mtx_ax
        self.cbar_ax = cbar_ax

        #Create Interactibles

        text_ax = fig.add_axes([0.03,0.03,0.03,0.03])
        plot_var_input = TextBox(text_ax,'Plot')
        plot_var_input.on_submit(self.change_plot_var)

        fig.canvas.mpl_connect('key_press_event', self.on_press)

        #Update plot
        #Figure Formatting

        pcs_ax.set_aspect('equal')
        cbar_ax.set_aspect(10)

        plt.tight_layout()

        plt.show()

class InteractiveFFTPlot:

    def __init__(self,data_dir,bgrid_file_loc,numpy_dir):

         # Collect grid quantities
        d = BoutData(path=data_dir)
        print(d["outputs"]["t_array"])
        self.nt = d["outputs"]["t_array"].shape[0] #int(d["outputs"]["Ni"].shape[0]) d["outputs"]["t"].shape[0]
        self.nx = 260#d["outputs"]["NXPE"] #int(d["outputs"]["Ni"].shape[1]) d["outputs"]["nx"]
        self.ny = 64 #d["outputs"]["NXPY"] #int(d["outputs"]["Ni"].shape[2]) d["outputs"]["ny"]
        self.nz = 65 #d["outputs"]["MZ"] #int(d["outputs"]["Ni"].shape[3]) d["outputs"]["nz"]
        print("nt nx ny nz = ",self.nt,self.nx,self.ny,self.nz)

        with DataFile(bgrid_file_loc) as g:
            self.rxy = g["Rxy"]
            self.zxy = g["Zxy"]
            #Bpxy = g["Bpxy"]
            #Btxy = g["Btxy"]

        #bout=BOUT_data(bgrid_file_loc)
        #self.rxy = bout.Rxy
        #self.zxy = bout.Zxy
        #self.nx,self.ny=bout.Rxy.shape # here (x,y)=(radial,poloidal)
        self.x = range(0,self.nx+1)
        self.y = range(0,self.ny+1)
        self.t = 0

        self.numpy_dir = numpy_dir

        self.cmap = 'cividis'
        self.norm = mc.TwoSlopeNorm(0)
        self.norm_type = 'linear'
        self.plot()

    def load(self,input_var):
        out = np.load(self.numpy_dir / (input_var+".npy"))
        print(out.shape)
        return out

    def execute_function(self):
        #print("ndim = ",self.ndim)
        if self.ndim == 4:
            fft_var = FFTN(self.var[self.t,:,:,:])
            out = fft_var
            print(out)
            print(out.shape)
        if self.ndim == 3:
            print('This var only has 3 dimensions')
        if self.ndim == 2:
            print('This var only has 2 dimensions')
        return out

    def update(self):
        #Execute function

        values = self.execute_function()
        val_min = np.min(values)
        #print("val_min = ",val_min)
        val_max = np.max(values)
        #print("val_max = ",val_max)
        if val_min >= 0:
            self.cmap = 'Blues'
            if self.norm_type == 'linear':
                self.norm = mc.Normalize(val_min,val_max)
            if self.norm_type == 'log':
                self.norm = mc.LogNorm(val_min,val_max)
        elif val_max <= 0:
            self.cmap = 'Blues_r'
            if self.norm_type == 'linear':
                self.norm = mc.Normalize(val_min,val_max)
            if self.norm_type == 'log':
                self.norm = mc.SymLogNorm(0.0001*val_min,1,val_min,val_max)
        else:
            self.cmap = 'cividis'
            if self.norm_type == 'linear':
                self.norm = mc.TwoSlopeNorm(0,val_min,val_max)
            if self.norm_type == 'log':
                self.norm = mc.SymLogNorm(0.0001*val_max,1,val_min,val_max)
        self.ax_XY.clear()
        self.ax_XZ.clear()
        self.ax_YZ.clear()

        #amp_x = values[:,:,30]
        #amp_y = values[:,30,]
        #amp_z =

        freq_x,freq_y,freq_z = np.meshgrid(FREQ(values.shape[1],),FREQ(values.shape[0]),FREQ(values.shape[2]))
        print(freq_x.shape)
        print(freq_y.shape)
        print(freq_z.shape)

        ax_XY_plot = self.ax_XY.plot_surface(freq_x[:,:,0],freq_y[:,:,0],values[:,:,0],cmap=self.cmap)
        ax_XZ_plot = self.ax_XZ.plot_surface(freq_x[:,48,:],freq_z[:,48,:],values[:,48,:],cmap=self.cmap)
        ax_YZ_plot = self.ax_YZ.plot_surface(freq_y[150,:,:],freq_z[150,:,:],values[150,:,:],cmap=self.cmap) 

        #pcs_ax_plot = self.pcs_ax.pcolor(self.rxy, self.zxy, values, cmap=self.cmap, norm=self.norm)
        #mtx_ax_plot = self.mtx_ax.pcolormesh(self.y, self.x, values, cmap=self.cmap, norm=self.norm, shading='auto')
        

        #if int(self.ndim) == 4:
        #    self.mtx_ax.set_title("t = %i   z = %i" % (self.t+1, self.z+1))
        self.ax_XY.set_title(str(self.t))
        #if int(self.ndim) == 2:
            #self.mtx_ax.set_title("Plasma Background")
        #self.cbar = plt.colorbar(mtx_ax_plot,cax=self.cbar_ax)
        self.fig.canvas.draw_idle()

    def change_plot_var(self,input_var):
        self.var = self.load(input_var)
        self.ndim = len(self.var.shape)
        self.ax_XY.clear()
        self.ax_XZ.clear()
        self.ax_YZ.clear()
        self.update()

    def on_press(self,event):

        if event.key == 'up':
            #increase z
            self.z += 1
            if self.z > self.nz-1:
                self.z = self.nz-1
            self.update()

        if event.key == 'down':
            #decrease z
            self.z -= 1
            if self.z < 0:
                self.z = 0
            self.update()

        if event.key == 'left':
            #decrease time
            self.t -= 1
            if self.t < 0:
                self.t = 0
            self.update()

        if event.key == 'right':
            #increase time
            self.t += 1
            if self.t > self.nt-1:
                self.t = self.nt-1
            self.update()

        if event.key == 'control':
            if self.norm_type == 'linear':
                self.norm_type == 'log'
            if self.norm_type == 'log':
                self.norm_type == 'linear'
            self.update()

    def plot(self):

        self.t = -1

        #Create Fig

        fig = plt.figure()
        ax_XY = fig.add_subplot(131, projection='3d')
        ax_XZ = fig.add_subplot(132, projection='3d')
        ax_YZ = fig.add_subplot(133, projection='3d')

        self.fig = fig
        self.ax_XY = ax_XY
        self.ax_XZ = ax_XZ
        self.ax_YZ = ax_YZ

        #Create Interactibles

        text_ax = fig.add_axes([0.03,0.03,0.03,0.03])
        plot_var_input = TextBox(text_ax,'Plot')
        plot_var_input.on_submit(self.change_plot_var)

        fig.canvas.mpl_connect('key_press_event', self.on_press)

        #Update plot
        #Figure Formatting

        #pcs_ax.set_aspect('equal')
        #cbar_ax.set_aspect(10)

        plt.tight_layout()

        plt.show()

if __name__ == "__main__":

    print("Call this function from another script such as 'bout_int_radial1d'")
    exit() 
