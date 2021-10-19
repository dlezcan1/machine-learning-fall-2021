import matplotlib.colors
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


def get_line(w, b, x_lim = (-np.inf, np.inf), y_lim = (-np.inf, np.inf)):
    w /= np.linalg.norm(w)

    x0 = w[0]*b
    y0 = w[1]*b

    t = 1000
    x_pt = x0 + t*np.array([-w[1], w[1]])
    y_pt = y0 + t*np.array([w[0], -w[0]])

    return x_pt, y_pt

# get_line

y = np.array([1,1,1,2,2,2,3,3,3])
X = np.array([
              [0,1],  [0,2],  [1,1],
              [-2,0], [-1,0], [-1,-1],
              [2,0],  [3,0],  [2,-1]
              ])

model = svm.SVC(kernel='linear')
model.fit(X,y)

Xg, Yg = np.meshgrid(np.linspace(-3,4,100), np.linspace(-3,4,100))
Cg = model.predict(np.c_[Xg.ravel(), Yg.ravel()]).reshape(Xg.shape)

print(model.classes_)
print(model.support_, model.n_support_)
print(model.support_vectors_)

# support lines
w = np.array([[0, 1, 1/2],
              [-1, -1, 0],
              [1, -1, 1]
             ])
support_vectors = [[0, 2, 3,4, 6,7], [0, 4], [2, 6]]

# fig, ax = plt.subplots()
#
# ax.contourf(Xg, Yg, Cg, cmap=plt.cm.coolwarm, alpha=0.8)
# ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#
# plt.show()

# plot lines
colors = ['r','g','b']
fig, ax = plt.subplots(figsize=(6,6))
colors_m = matplotlib.colors.ListedColormap(colors)
ax.scatter(X[:,0], X[:,1], c=y-1, cmap=colors_m)
for i in range(w.shape[0]):
    w_i = w[i,0:2]
    b_i = w[i,2]
    x_i, y_i = get_line(w_i, b_i)
    ax.plot(x_i, y_i, colors[i])

    marker_size = 200 if i == 0 else 100
    idxs = support_vectors[i]
    svs = X[idxs]
    ax.scatter(svs[:,0], svs[:,1], c="none",  edgecolor=colors[i], s=len(idxs)*[marker_size], cmap=colors_m)
    # for idx in support_vectors[i]:
    #     vect = X[idx]
    #     ax.scatter(vect[0], vect[1], c=[i], cmap=colors_m, s=[marker_size])

# for

ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])


plt.show()