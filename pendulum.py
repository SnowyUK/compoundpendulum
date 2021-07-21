""""
Simulates a double pendulum. This code is extensively borrowed from https://scipython.com/blog/the-double-pendulum/
"""

import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import argparse
import datetime

def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) - (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def calc_E(y):
    """Return the total energy of the system."""

    th1, th1d, th2, th2d = y.T
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V

def make_plot(i):
    # Plot and save an image of the double pendulum configuration for time
    # point i.
    # The pendulum rods.
    ax.plot([0, xe1[i], xe2[i]], [0, ye1[i], ye2[i]], lw=2, c='dimgrey')
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
    # Circles representing the anchor point of rod 1, and bobs 1 and 2.
    c0 = Circle((0, 0), r1/2, fc='k', zorder=10)
    c1 = Circle((x1[i], y1[i]), r1, fc='b', ec='b', zorder=10)
    c2 = Circle((x2[i], y2[i]), r2, fc='r', ec='r', zorder=10)
    c3 = Circle((xe1[i], ye1[i]), r1, fc='b', ec='b', zorder=10)
    c4 = Circle((xe2[i], ye2[i]), r2, fc='g', ec='g', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)
    ax.add_patch(c4)

    # The trail will be divided into ns segments and plotted as a fading line.
    ns = 20
    s = max_trail // ns

    for j in range(ns):
        imin = i - (ns-j)*s
        if imin < 0:
            continue
        imax = imin + s + 1
        # The fading looks better if we square the fractional length along the
        # trail.
        alpha = (j/ns)**2
        ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt', lw=2, alpha=alpha)
        ax.plot(xe2[imin:imax], ye2[imin:imax], c='g', solid_capstyle='butt', lw=2, alpha=alpha)

    # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-args.l1-args.l2-r1, args.l1+args.l2+r1)
    ax.set_ylim(-args.l1-args.l2-r1, args.l1+args.l2+r1)
    ax.set_aspect('equal', adjustable='box')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    plt.title("Compound Pendulum")
    plt.grid()
    plt.savefig(f'frames/_img{i//di:04d}.png', dpi=args.dpi)
    plt.cla()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Simulation of compound pendulum", epilog="SnowyUK 2021")
    ap.add_argument("--output", metavar="directory", default=f"frames_{datetime.datetime.now():'%Y-%m-%d_%h_%m_%s'}")
    ap.add_argument("--nuke", action="store_true", help="Delete any files in output directory")
    ap.add_argument("--duration", metavar="seconds", type=float, help="Duration of animation", default=60.0)
    ap.add_argument("--l1", metavar="m", type=float, default=1.0, help="Length of first rod")
    ap.add_argument("--l2", metavar="m", type=float, default=1.0, help="Length of second rod")
    ap.add_argument("--m1", metavar="kg", type=float, default=1.0, help="Mass of first weight")
    ap.add_argument("--m2", metavar="kg", type=float, default=1.0, help="Mass of second weight")
    ap.add_argument("--dt", metavar="s", type=float, default=0.01, help="Time point spacings")
    ap.add_argument("--et1", metavar="value", type=float, default=1e-6, help="Error in \N{Greek small letter theta}1")
    ap.add_argument("--et2", metavar="value", type=float, default=0.000, help="Error in \N{Greek small letter theta}2")
    ap.add_argument("--fps", metavar="n", type=int, default=20, help="Number of video frames per second")
    ap.add_argument("--dpi", metavar="n", type=int, default=150, help="Dots per inch for PNG output")
    args = ap.parse_args()
    # Pendulum rod lengths (m), bob masses (kg).
    g = 9.81

    # Maximum time, time point spacings and the time grid (all in s).
    tmax, dt = args.duration, args.dt
    t = np.arange(0, tmax+dt, dt)
    # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
    y0 = np.array([np.pi, 0, 7*np.pi/8, 0])
    ye0 = np.array([np.pi*(1+args.et1), 0, 7*np.pi*(1+args.et2)/8, 0])


    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(args.l1, args.l2, args.m1, args.m2))
    ye = odeint(deriv, ye0, t, args=(args.l1, args.l2, args.m1, args.m2))

    # Unpack z and theta as a function of time
    theta1, theta2 = y[:, 0], y[:, 2]
    thetae1, thetae2 = ye[:, 0], ye[:, 2]

    # Convert to Cartesian coordinates of the two bob positions.
    x1 = args.l1 * np.sin(theta1)
    y1 = -args.l1 * np.cos(theta1)
    x2 = x1 + args.l2 * np.sin(theta2)
    y2 = y1 - args.l2 * np.cos(theta2)

    xe1 = args.l1 * np.sin(thetae1)
    ye1 = -args.l1 * np.cos(thetae1)
    xe2 = x1 + args.l2 * np.sin(thetae2)
    ye2 = y1 - args.l2 * np.cos(thetae2)

    # Plotted bob circle radius
    r1 = 0.05*np.sqrt(args.m1)
    r2 = 0.05*np.sqrt(args.m2)

    # Plot a trail of the m2 bob's position for the last trail_secs seconds.
    trail_secs = 1
    # This corresponds to max_trail time points.
    max_trail = int(trail_secs / dt)


    # Make an image every di time points, corresponding to a frame rate of fps
    # frames per second.
    # Frame rate, s-1
    fps = args.fps
    di = int(1/fps/dt)
    fig = plt.figure(figsize=(8.3333, 6.25), dpi=args.dpi)
    ax = fig.add_subplot(111)

    for i in range(0, t.size, di):
        print(i // di, '/', t.size // di)
        make_plot(i)
