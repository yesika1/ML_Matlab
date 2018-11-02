#Histogram.py


for c in df3.columns:
    df3.loc[:,c].hist()
    plt.title(c + ' Histogram')
    plt.show()


 def histograms(data):
    for c in data.columns:
        if (data[c].dtype != object):
            df3.loc[:,c].hist()
            plt.title(c + ' Histogram')
            plt.show()

 histograms(df3)