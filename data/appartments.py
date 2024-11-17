import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

def plot_decay( decay_rate = 0.2):
  x = np.arange(1, 21)
  y = np.exp(-decay_rate * x)

  plt.plot(x, y)
  plt.xlabel('District')
  plt.ylabel('Proportion')
  plt.title('District Premium Ratio (Using Exponential Decay)')
  plt.grid(True)
  plt.show()

def plot_appartments():
  
  def thousands_formatter(x, pos):
      return f'{int(x/1000)}M'

  sns.scatterplot(data=df, x=df.m2, y=df.price, hue=df.district, style=df.balconies_count ,size=df.cave_m2 ,  palette='tab20', sizes=(10, 100), alpha=0.9)

  plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=4)