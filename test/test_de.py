import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint, solve_ivp

import numba


def first_attempt():
    # ode right-hand-side
    def dxdt(t, x):
        return - x * x
    # initial condition
    x0 = 1

    t = np.linspace(0., 1., 100)

    sol_meth1 = odeint(dxdt, y0=x0, t=t, tfirst=True)
    sol_meth2 = solve_ivp(dxdt, [0, np.max(t)], [x0], t_eval=t)

    y1 = sol_meth1.T[0]
    y2 = sol_meth2.y[0]

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(t, y1, c='red', label='meth1')
    axs[0].plot(t, y2, c='blue', label='meth2')
    axs[1].plot(t, y1 - y2, c='green', label='difference')
    axs[0].legend()
    axs[1].legend()
    plt.show()
    plt.close(fig)


def first_system():
    pass


# def RK45(f, a, b, y0,
#          h, hmin, hmax, e, Itermax, ):
#
#     x = a
#     y = y0
#
#     hmax = h
#     hmin = 1.e-7
#
#     if b - a > h:
#         k1 = h * f(x, y)
#         k2 = h * f(x + h * 0.25, y + k1 * 0.25)
#         k3 = h * f(x + (3/8)*h, y + (3/32)*k1 + (9/32)*k2)
#         k4 = h * f(x + (12/13)*h, y + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
#         k5=h*f(x+h,y+((ld)439/216)*k1-8*k2+((ld)3680/513)*k3-((ld)845/4104)*k4);
#         k6=h*f(x+h*((ld)1/2),y-((ld)8/27)*k1+2*k2-((ld)3544/2565)*k3+((ld)1859/4104)*k4-((ld)11/40)*k5);
#
#     pass

# ld RK45(ld f(ld,ld),  //ф-ия f(x,y)
# 	 ld y0,    //y(a)=y0-нач условие
# 	 ld a,ld b,//b-точка в кот считается знач ф-ии
# 	 ld h, //начальный шаг
# 	 ld hmin,//min шаг (если =0 => Auto)
# 	 ld hmax,//max шаг (=0 => (b-a)/20 )
# 	 ld e,//погрешность на шаге
# 	 long int Itermax,//max число итераций
# 	 char *c)//=0-OK
# 		 //=-1 если кол-во итераций>Itermax
# { ld x,y,k1,k2,k3,k4,k5,k6,r,r1,r2;
#   long int i;
#   x=a; y=y0;
# if (hmax==0)  hmax=(b-a)/20;
# if (hmin==0) //считаем машинное eps
#    {r=fabsl(a)>fabsl(b)?fabsl(a):fabsl(b);
#     for(hmin=1;(hmin+r)!=r;hmin/=2);
#     hmin*=4;
#    }
# if (b-a>=h)
#   {do
#      {k1=h*f(x,y);
#       k2=h*f(x+h*((ld)1/4),y+k1*((ld)1/4));
#       k3=h*f(x+((ld)3/8)*h,y+((ld)3/32)*k1+((ld)9/32)*k2);
#       k4=h*f(x+((ld)12/13)*h,y+((ld)1932/2197)*k1-((ld)7200/2197)*k2+((ld)7296/2197)*k3);
#       k5=h*f(x+h,y+((ld)439/216)*k1-8*k2+((ld)3680/513)*k3-((ld)845/4104)*k4);
#       k6=h*f(x+h*((ld)1/2),y-((ld)8/27)*k1+2*k2-((ld)3544/2565)*k3+((ld)1859/4104)*k4-((ld)11/40)*k5);
#
#       r1=((ld)16/135)*k1+((ld)6656/12825)*k3+((ld)28561/56430)*k4-((ld)9/50)*k5+((ld)2/55)*k6;
#       r2=((ld)25/216)*k1+((ld)1408/2565)*k3+((ld)2197/4104)*k4-k5*((ld)1/5);
#       r=fabsl(r1-r2);
#
#       if (r>e)
# 	{if (h>hmin) h/=2;
# 	 else {x+=h; y+=r1;}
# 	}
#       else
# 	{x+=h; y+=r1;
# 	 if ((r<e*0.02)&&(h<hmax)) h*=2;
# 	}
#
#       if (b-x<h) h=b-x;
#      i++;
#      }
#    while ((x<b)&&(i<=Itermax));
#   }
#
# if (i>Itermax) *c=-1;
#    else *c=0;
# return y;
# };


@numba.njit
def RK45vec_step(F, t0, y0, h):

    t = t0
    y = y0

    f = F(t, y)
    k1 = h * f
    r = y + k1 / 4
    r1 = (16/135) * k1
    r2 = (25/216) * k1

    f = F(t + h / 4, r)
    k2 = h * f
    r = y + (3/32)*k1 + (9/32)*k2

    f = F(t+(3/8) * h, r)
    k3 = h * f
    r = y + (1932./2197.)*k1 - (7200./2197.)*k2 + (7296./2197.)*k3
    r1 += (6656./12825.)*k3
    r2 += (1408./2565.)*k3

    f = F(t+(12/13) * h, r)
    k4 = h * f
    r = y + (439./216.) * k1 - 8*k2 + (3680./513.)*k3 - (845./4104.)*k4
    r1 += (28561./56430.)*k4
    r2 += (2197./4104.)*k4

    f = F(t + h, r)
    k5 = h * f
    r = y - (8/27)*k1 + 2*k2 - (3544./2565.)*k3 + (1859./4104.)*k4 - (11/40)*k5
    r1 -= (9/50)*k5
    r2 -= k5/5

    f = F(t+h/2, r)
    k6 = h * f
    r1 += (2/55)*k6

    # if np.sum(np.abs(r1 - r2)) > e:
    #     pass

    t += h
    y += r1

    err = np.sum(np.abs(r1 - r2))

    return t, y, err


def test_numba():

    @numba.njit
    def func(F, x1, x2):
        return F(x1), F(x2)

    @numba.njit
    def func_to_func(x):
        return x[0] ** 2 + x[1]

    # test with external objects
    # this works
    # b = np.array([1, 4])
    #
    # @numba.njit
    # def func_to_func_2(x):
    #     return x * b

    # this doesn't work!
    # @numba.njit
    # def get_func(b: np.ndarray):
    #
    #     @numba.njit
    #     def f(x):
    #         x * b
    #
    #     return f

    # this works
    def get_func(b: np.ndarray):

        @numba.njit
        def f(x):
            return x * b

        return f

    ret1, ret2 = func(get_func(np.array([1, 4])), np.array([2, 3]), np.array([3, 2]))

    print(ret1, ret2)


# void RK45Syst(char n,//кол-во уравнений в нормальной системе(mmaxMass-max)
# 	    void F(ld,ld*,ld*),  //ф-ии f1(x,y1..yn)...fn
# 	    ld* y0,    //y(a)=y0-нач условие
# 	    ld* y1,//возвращаемое знач
# 	    ld a,ld b,//a=t0,b-точка в кот считается знач ф-ии
# 	    ld h, //начальный шаг
# 	    ld hmin,//min шаг (если =0 => Auto)
# 	    ld hmax,//max шаг (=0 => (b-a)/20 )
# 	    ld e,//погрешность на шаге
# 	    long int Itermax,//max число итераций
# 	    char *c)//=0-OK
# 		    //=-1 если кол-во итераций>Itermax
# { ld f[mmaxMass],x,y[mmaxMass],norma,k1[mmaxMass],k2[mmaxMass],k3[mmaxMass],k4[mmaxMass],k5[mmaxMass],k6[mmaxMass];
#   ld r[mmaxMass],r1[mmaxMass],r2[mmaxMass],rr;
#   char i;
#   long int j=0;
#
# x=a;
# For y[i]=y0[i];
#
# if (hmax==0)  hmax=(b-a)/20;
# if (hmin==0) //считаем машинное eps
#    {rr=fabsl(a)>fabsl(b)?fabsl(a):fabsl(b);
#     for(hmin=1;(hmin+rr)!=rr;hmin/=2);
#     hmin*=4;
#    }
# if (b-a>=h) h=b-a;
# if (h<=0) {*c=-1; return;}
#   do
#      {F(x,y,f);
#       For {k1[i]=h*f[i];          r[i]=y[i]+k1[i]*((ld)1/4);
# 	   r1[i]=((ld)16/135)*k1[i];  r2[i]=((ld)25/216)*k1[i]; }
#       F(x+h*((ld)1/4),r,f);
#       For {k2[i]=h*f[i];  r[i]=y[i]+((ld)3/32)*k1[i]+((ld)9/32)*k2[i]; }
#       F(x+((ld)3/8)*h,r,f);
#       For {k3[i]=h*f[i];  r[i]=y[i]+((ld)1932./2197.)*k1[i]-((ld)7200./2197.)*k2[i]+((ld)7296./2197.)*k3[i];
# 	  r1[i]+=((ld)6656./12825.)*k3[i]; r2[i]+=((ld)1408./2565.)*k3[i]; }
#       F(x+((ld)12/13)*h,r,f);
#       For {k4[i]=h*f[i]; r[i]=y[i]+((ld)439./216.)*k1[i]-8*k2[i]+((ld)3680./513.)*k3[i]-((ld)845./4104.)*k4[i];
# 	   r1[i]+=((ld)28561./56430.)*k4[i]; r2[i]+=((ld)2197./4104.)*k4[i]; }
#       F(x+h,r,f);
#       For {k5[i]=h*f[i]; r[i]=y[i]-((ld)8/27)*k1[i]+2*k2[i]-((ld)3544./2565.)*k3[i]+((ld)1859./4104.)*k4[i]-((ld)11/40)*k5[i];
# 	  r1[i]-=((ld)9/50)*k5[i]; r2[i]-=k5[i]/5; }
#       F(x+h*((ld)0.5),r,f);
#       For {k6[i]=h*f[i]; r1[i]+=((ld)2/55)*k6[i]; }
#
#       norma=0;
#       For {norma+=fabsl(r1[i]-r2[i]);}
#
#
#       if (norma>e)
# 	{if (h>hmin) h/=2;
# 	 else {x+=h; For y[i]+=r1[i];}
# 	}
#       else
# 	{x+=h; For y[i]+=r1[i];
# 	 if ((norma<e*0.02)&&(h<hmax)) h*=2;
# 	}
#
#       if (b-x<h) h=b-x;
#      j++;
#      }
#    while ((x<b)&&(j<=Itermax));
# if (j>Itermax) *c=-1;
#    else *c=0;
# For y1[i]=y[i];
# }


def main():
    # first_attempt()
    test_numba()
    pass


if __name__ == '__main__':
    main()
