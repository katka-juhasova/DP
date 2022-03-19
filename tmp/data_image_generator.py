import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
other_images = ['../docs/images/data-samples/other1.png',
                '../docs/images/data-samples/other2.png',
                '../docs/images/data-samples/other5.png',
                '../docs/images/data-samples/star.png',
                '../docs/images/data-samples/other6.png'
                ]

fig = plt.figure(figsize=(8, 2))

for i, mess in enumerate(other_images):
    img = mpimg.imread(mess)

    ax = plt.subplot(1, len(other_images), i + 1)
    ax.imshow(img, cmap='gray')
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('3')

fig.subplots_adjust(wspace=0.1, hspace=0)
fig_path = os.path.join(BASE_DIR, '..', 'docs', 'images', 'data-other.png')
plt.savefig(fig_path)
# plt.show()

symbol_images = ['../docs/images/data-samples/A.png',
                 '../docs/images/data-samples/t.png',
                 '../docs/images/data-samples/G.png',
                 '../docs/images/data-samples/S2.png',
                 '../docs/images/data-samples/O.png',
                 '../docs/images/data-samples/2.png',
                 '../docs/images/data-samples/5.png',
                 '../docs/images/data-samples/7.png',
                 '../docs/images/data-samples/8.png',
                 '../docs/images/data-samples/9.png']

fig = plt.figure(figsize=(7, 2))

for i, sign in enumerate(symbol_images):
    img = mpimg.imread(sign)

    ax = plt.subplot(2, 5, i + 1)
    ax.imshow(img, cmap='gray')
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('3')

fig.subplots_adjust(wspace=0, hspace=0.2)
fig_path = os.path.join(BASE_DIR, '..', 'docs', 'images', 'data-symbols.png')
plt.savefig(fig_path)
# plt.show()
