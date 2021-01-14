import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 7.5];#default plot size
plt.rcParams['font.size']=16;
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['savefig.facecolor']='white'
from sympy.solvers import solve
import sympy as sp
#----------------------------------------------------------------------------
#........................... ELASTIC LOADING...............................
#------------------------------------------------------------------------
directories=[
r'D:\3View\9am-achilles-fshx\7.5um_crop_2\dummy_110_124\results',
r'D:\3View\9am-achilles-fshx\7.5um_crop_2\dummy_100_200\results',
r'D:\3View\7pmAchx700\7-5um\dummy_0_100\results']
DSN=1;#dataset number
resultsDirectory=directories[DSN];

elasticmodulus = 350; #MPa VanDerRijt2006
e=np.linspace(0.0,0.075,1000); #strain values
vol_frac=0.706; #from Ben - fibril fraction in the image

def elastic_contribution(DSN):
    """
    Generates a list of points based on e
    DSN= Dataset number 0=9am, 1=7pm
    """
    try:
        e_c=np.load(directories[DSN]+r'\scaledlengths.npy')-1; #critical strain
        radii=np.load(directories[DSN]+r'\radii.npy');
        areas=np.load(directories[DSN]+r'\area.npy');

    except:
        print("No fibril record found")

    nfibs=e_c.size;

    def F_fib(i, e): #e=global strain, given in nN
        f=elasticmodulus*areas[i]*(e-e_c[i])/(1+e_c[i]);
        if f>0:
            return f;
        else:
            return 0;
    def stress(e): #in GPa
        F=0;
        for i in range (nfibs):
            F+=F_fib(i,e);
        return F/np.sum(areas);

    s=[stress(i) for i in e];
    return e_c, s;


#Criticalstrain
e_c=np.array(elastic_contribution(DSN)[0]);
n, bins, patches = plt.hist(100*e_c, 50, density=False, facecolor='g', alpha=0.75)
plt.xlabel('Critical strain (%)')
plt.ylabel('Number')
plt.grid(True)
plt.savefig(resultsDirectory+'\critStrain9am');
plt.show()




#%%plot the elastic stress strain curve
elastic_stress=np.array(elastic_contribution(DSN)[1]);
plt.plot(100*e,elastic_stress, '-r');
plt.xlabel('Strain (%)');
plt.ylabel('Stress (MPa)');
plt.savefig(resultsDirectory+'\stressStrain9am');
plt.show()
#%%----------------------------------------------------------------------------
#........................... CHOOSE STRAIN RATE.....................
#------------------------------------------------------------------------

B, C, D ,F= [np.ptp(e)/2,10/(2*sp.pi),-sp.pi/2, np.mean(e)];

def e_t(B, C, D, F, t):
    return B *sp.sin( C*t+D) + F;

t=sp.symbols('t')
y=e_t(B, C, D, F, t);
lam_y = sp.lambdify(t, 100*y, modules=['numpy']);
t_vals = np.linspace(0, 10, 100);
y_vals = lam_y(t_vals);
plt.plot(t_vals, y_vals);
plt.xlabel("Time (s)")
plt.ylabel("Strain (%)")
plt.savefig(resultsDirectory+'\imposedStrain9am');
plt.show()



#%%----------------------------------------------------------------------------
#........................... HYSTERESIS CURVE.....................
#------------------------------------------------------------------------

fluidstress=0.04*1000#0.0000139;#543.974; MPA

def find_timepoints_when_singular(timepoints):
    pad = len(max(timepoints, key=len))
    timepoints=np.array([i + [0]*(pad-len(i)) for i in timepoints])
    meetwhere=np.argwhere(timepoints[:,1]==0)[0,0]
    timepoints[meetwhere,1]=timepoints[meetwhere,0]
    return timepoints;

t=sp.symbols('t');
timepoints=[solve(strain - e_t(B,C,D,F,t), t) for strain in e];

if np.any(np.array([len(xi) for xi in timepoints])==1): #Singular
    timepoints=find_timepoints_when_singular(timepoints);

def deps_dt(B, C, D, tt):
    return sp.diff(e_t(B,C,D,F, t),t).subs(t, tt)

elastic_stress=np.array(elastic_contribution(DSN)[1]);
viscoelastic_stress_loading=np.array([fluidstress*sp.N(deps_dt(B, C, D, tt) )for tt in timepoints[:,0]]);
viscoelastic_stress_unloading=np.array([fluidstress*sp.N(deps_dt(B, C, D, tt) )for tt in timepoints[:,1]]);
totalstress_loading=vol_frac*elastic_stress+(1-vol_frac)*viscoelastic_stress_loading;
totalstress_unloading=vol_frac*elastic_stress+(1-vol_frac)*viscoelastic_stress_unloading;


plt.plot(100*e,totalstress_loading, label='Loading');
plt.plot(100*e,totalstress_unloading,label='Unloading');
plt.xlabel('Strain (%)')
plt.ylabel('Stress (MPa)')
plt.legend(loc="upper left")
plt.savefig(resultsDirectory+'\hysteresis9am');
plt.show()


#%%----------------------------------------------------------------------------
#...........................TEXT FILE RESULTS....................
#------------------------------------------------------------------------
headings=['fluid stress (chosen)', 'elastic modulus (Gpa)', 'B', 'C', 'D', 'F']
valuessymb=[fluidstress,elasticmodulus, B, C, D, F]

values=[sp.N(xi) for xi in [fluidstress,elasticmodulus, B, C, D, F]]



import csv
with open(resultsDirectory+'\params.csv', mode='w') as params:
    params = csv.writer(params, delimiter=',')
    params.writerow(headings)
    params.writerow(valuessymb)
    params.writerow(values)
