import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
other_images = ['tmp/data-samples/other1.png',
                'tmp/data-samples/other2.png',
                'tmp/data-samples/other5.png',
                'tmp/data-samples/star.png',
                'tmp/data-samples/other6.png'
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
fig_path = os.path.join(BASE_DIR, '..', 'docs', 'data-other.png')
plt.savefig(fig_path)
# plt.show()

symbol_images = ['tmp/data-samples/A.png',
                 'tmp/data-samples/t.png',
                 'tmp/data-samples/G.png',
                 'tmp/data-samples/S2.png',
                 'tmp/data-samples/O.png',
                 'tmp/data-samples/2.png',
                 'tmp/data-samples/5.png',
                 'tmp/data-samples/7.png',
                 'tmp/data-samples/8.png',
                 'tmp/data-samples/9.png']

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
fig_path = os.path.join(BASE_DIR, '..', 'docs', 'data-symbols.png')
plt.savefig(fig_path)
# plt.show()
