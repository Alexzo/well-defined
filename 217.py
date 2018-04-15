from pylab import *
from mpl_toolkits.mplot3d import Axes3D
def g(w):
    # return (2.0*exp(dot(w.T, w))*w)/(1.0+exp(dot(w.T, w)))
    j= (2.0*exp(dot(w.T, w))*(2*dot(w.T, w)+exp(dot(w, w.T))+1.0))/((1+exp(dot(w.T, w)))**2)
    return dot(pinv(j),(2.0*exp(dot(w.T, w))*w)/(1.0+exp(dot(w.T, w))))
    # return ((2.0*exp(dot(w.T, w))*w)/(1.0+exp(dot(w.T, w)))/((4.0 * exp(-dot(w.T, w)) * dot(w, w.T)) / (1.0 + exp(-dot(w.T, w))) ** 2))
w0 = array([1,1])
w0.shape=(2,1)
print(log(1+exp(dot(w0.T, w0))))
print(g(w0))
w0=w0-g(w0)
print(log(1+exp(dot(w0.T, w0))))
# def make_function():
#     global fig,ax1
#
#     # prepare the function for plotting
#     r = linspace(-1.15,1.15,300)
#     s,t = meshgrid(r,r)
#     s = reshape(s,(size(s),1))
#     t = reshape(t,(size(t),1))
#     h = concatenate((s,t),1)
#     h = dot(h*h,ones((2,1)))
#     b = log(1+exp(h))
#     s = reshape(s, (int(sqrt(size(s))), int(sqrt(size(s)))))
#     t = reshape(t, (int(sqrt(size(t))), int(sqrt(size(t)))))
#     b = reshape(b, (int(sqrt(size(b))), int(sqrt(size(b)))))
#
#     # plot the function
#     fig = plt.figure(facecolor = 'white')
#     ax1 = fig.add_subplot(111, projection='3d')
#     ax1.plot_surface(s,t,b,cmap = 'Greys',antialiased=False) # optinal surface-smoothing args rstride=1, cstride=1,linewidth=0
#     ax1.azim = 115
#     ax1.elev = 70
#
#     # pretty the figure up
#     ax1.xaxis.set_rotate_label(False)
#     ax1.yaxis.set_rotate_label(False)
#     ax1.zaxis.set_rotate_label(False)
#     ax1.get_xaxis().set_ticks([-1,1])
#     ax1.get_yaxis().set_ticks([-1,1])
#     ax1.set_xlabel('$w_0$   ',fontsize=20,rotation = 0,linespacing = 10)
#     ax1.set_ylabel('$w_1$',fontsize=20,rotation = 0,labelpad = 50)
#     ax1.set_zlabel('$g(\mathbf{w})$',fontsize=20,rotation = 0,labelpad = 20)
#     show()

def gradient_descent(w0):
    w = w0
    g_path = []
    w_path = []
    w_path.append(w)
    g_path.append(log(1+exp(dot(w.T, w))))

    # start gradient descent loop
    grad = 1
    iter = 1
    max_its = 10
    while iter <= max_its:
        # take gradient step
        grad= dot(pinv((2.0*exp(dot(w.T, w))*(2*dot(w.T, w)+exp(dot(w, w.T))+1.0))/((1+exp(dot(w.T, w)))**2)),(2.0*exp(dot(w.T, w))*w)/(1.0+exp(dot(w.T, w))))
        w = w - grad

        # update path containers
        w_path.append(w)
        g_path.append(log(1+exp(dot(w.T, w))))
        iter+= 1


    return g_path

k=linspace(0,10,11)
cost=gradient_descent(w0)
costh=[item[0][0] for item in cost]
print(costh)
fig,ax =subplots(1,1,figsize=(6,6))
ax.plot(k,costh)

