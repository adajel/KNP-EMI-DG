from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

nx = 1500
len = 305e-3 # 305 um
mesh = IntervalMesh(nx, 0, len)

V = FunctionSpace(mesh, "CG", 1)
u_PDE = TrialFunction(V)
v = TestFunction(V)

w = Function(V)

# initial states
u_prev = Function(V)
m_prev = Function(V)
n_prev = Function(V)
h_prev = Function(V)

u_0 = Function(V)
u_1 = Function(V)
u_2 = Function(V)
u_3 = Function(V)
u_4 = Function(V)
u_5 = Function(V)
u_6 = Function(V)
u_7 = Function(V)

t = Constant(0)
dt = 0.01           # ms
Tstop = 10.0        # ms

# initial membrane potential (mV)
u_prev.assign(Constant(-54))

w_ = 6.0e-5         # geometric parameter (cm)
sigma_i = 20.12     # ICS conductance mS/cm
delta = (sigma_i * w_) / 4

# Membrane properties
g_leak = 0.4
C_M = 1.0
E_K = -88.98
E_Na = 54.81
E_leak = -54

lam = (w_ * sigma_i)/(4 * g_leak)
lam_ = sqrt(lam)

# Set stimulus PDE
g_syn_bar = 1000
#g_syn = Expression('g_syn_bar * exp(-fmod(t, 20.0)/2.0) * (x[0] < 20.0e-4)', \
                   #t=t, g_syn_bar=g_syn_bar, degree=4)
g_syn = Expression('g_syn_bar * (x[0] < 10.0e-4)', \
                   g_syn_bar=g_syn_bar, degree=4)

I_ion = g_leak * (u_prev - E_leak) - g_syn

a = C_M / dt * inner(u_PDE, v) * dx + delta * inner(grad(u_PDE), grad(v)) * dx
L = C_M / dt * u_prev * v * dx - I_ion * v * dx

phi_M = []
time = []

p1 = 20
p2 = 40
t1 = 0
t2 = 0

for k in range(int(round(Tstop/dt))):

    # solve PDE system
    solve(a == L, w)
    u_prev.assign(w) # update u

    if float(t) == 0:
        u_0.assign(w)
        print("0")
    elif 10 <= float(t) <= 11:
        u_1.assign(w)
        print("1")
    elif 20 <= float(t) <= 21:
        u_2.assign(w)
        print("2")
    elif 30 <= float(t) <= 31:
        u_3.assign(w)
        print("3")
    elif 40 <= float(t) <= 41:
        u_4.assign(w)
        print("4")

    # save for plot
    point = 10.0e-4
    phi_M.append(u_prev(point))
    time.append(float(t))

    if (u_prev(p1*1.0e-4) > 20 and t1 == 0):
        t1 = float(t)
    if (u_prev(p2*1.0e-4) > 20 and t2 == 0):
        t2 = float(t)

    # update time
    t.assign(t + dt)
    g_syn.t = t

#header = "x y1"
#phi_dat = [phi_M]
##time = np.arange(0, Tstop, dt)
#phi_dat.insert(0, time)
#a_phi = np.asarray(phi_dat)
#np.savetxt("../results/figures/phi_time_cable.dat", np.transpose(a_phi), delimiter=" ", header=header)

#delta_t = t2 - t1
#delta_x = p2 - p1

#print("velocity (um/ms) = ", delta_x/delta_t)

V_max = 40
print(lam_)

plt.figure()
#plot(u_0, label="init")
#plot(u_1, label="1 s")
#plot(u_2, label="5 s")
#plot(u_3, label="9 s")
#plot(u_4, label="12 s")
#plot(project(Constant(V_max/2.71828), V), label="lam")
plot(project(Constant(-54), V), label="lam")
plot(w, label="end time")
plt.axvline(x = lam_)
plt.ylim(-100, 50)
plt.legend()
plt.savefig("phi_space_passive.png")
plt.close()

# get x point where V is 1/e of V_max
V_max = max(w.vector())
V_min = min(w.vector())
V_diff = V_max - V_min
print("V_diff: ", V_diff)
print("lam_", lam_)

dx = 0.1e-4

for x in np.arange(0, len, dx):
    e = 2.718
    tmin = V_diff/(e) + V_min - 0.05
    tmax = V_diff/(e) + V_min + 0.05
    if tmin < w(x) < tmax:
        print("x:", x, "potential: ", w(x))

print("lam_", lam_)
print("tmax", tmax)
print("tmin", tmin)

def plot_membrane_space(fname, n):

    plt.figure(figsize=(4.6, 3.3))
    plt.xlabel(r"x-position ($\mu$m)")
    plt.ylabel("$\phi_M$ (mV)")
    plt.ylim([-80, -60])
    #plt.ylim([-100e-3, -70e-3])
    #plt.yticks([-80, -60, -40, -20, 0, 20, 40])
    #plt.ylim([-90, 50])

    colors_n = [c_1, c_2, c_3]

    header = "x r0 r1 r2"
    phi_dat = []

    for idx, n in enumerate([1, 50, 99]):
        phi_M_n = get_plottable_gamma_function(fname, n)
        n_new = n/10. # convert n to ms
        x = np.linspace(5, 305, len(phi_M_n))
        plt.plot(x, phi_M_n, linewidth=2.5, color=colors_n[idx], label=r"%d ms" % n_new)
        phi_dat.append(phi_M_n)

    plt.legend()
    #plt.tight_layout()
    # save figure
    plt.savefig("results/figures/phi_M_space_passive.png")
    plt.close()

    phi_dat.insert(0, x)
    a_phi = np.asarray(phi_dat)
    np.savetxt("results/figures/phi_space_passive.dat", np.transpose(a_phi), delimiter=" ", header=header)

    return

plt.figure()
plt.plot(phi_M)
plt.savefig("phi_time_passive.png")
plt.close()
print(phi_M[-1])
print(lam_)
