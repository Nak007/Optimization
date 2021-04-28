'''
Author: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-04-2021
'''
import pandas as pd, numpy as np, collections, time
import matplotlib.pyplot as plt
from itertools import product, combinations
import ipywidgets as widgets
from IPython.display import display

__all__  = ['impurity_binned_2d']

class impurity_binned_2d():
    
    '''
    Using the so-called "Gini Impurity", the algorithm 
    computes binning from two sets of data in order to
    minimize (optimize) the overall impurity.
    
    versionadded:: 30-04-2021
    
    Parameters
    ----------
    n_bins : `int`, optional, default: 20
    \t Sets of randomly created bin edges.
        
    max_iter : `int`, optional, default: 10
    \t Maximum number of iterations of the algorithm 
    \t for a single run.
    
    tol : `float`, optional, default: 1e-4
    \t Relative tolerance of the gini impurity between 
    \t two consecutive iterations to declare convergence.
    
    bin_width : `str`, optional, default: "percentile"
    \t Method for binning. If "equal", equal-width 
    \t binning is implemented, whereas "percentile",
    \t equal-sample binning with unique bin edges is used.

    Attributes
    ----------
    info : `pd.DataFrame` object
    \t Information table that contains output from  
    \t each iteration and is comprised of:
    \t - 'bin1'     : Number of `x1` bins 
    \t - 'bin2'     : Number of `x2` bins
    \t - 'gini'     : Gini impurity
    \t - 'add_bin1' : Added bin edge of `x1`
    \t - 'add_bin2' : Added bin edge of `x2`
    \t - 'pct_delta': Relative difference of Gini

    bins_ : `collections.OrderedDict` object
    \t An ordered dictionary that keeps the order of
    \t the keys. It contains named tuples, namely 
    \t "RandBins" with positions as follows:
    \t - 'combi'     : Number of bins of `x1`, and `x2`
    \t - 'gini'      : Weighted Gini impurity
    \t - 'bin_edges1': Bin edges of `x1`  
    \t - 'bin_edges2': Bin edges of `x2`
    \t - 'add_bin1'  : Added bin edge of `x1`
    \t - 'add_bin2'  : Added bin edge of `x2`

    Examples
    --------
    >>> from Optimization import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_breast_cancer
    
    >>> X = pd.DataFrame(load_breast_cancer().data)
    >>> X.columns = [c.replace(' ','_') for c in 
    ...              load_breast_cancer().feature_names]

    >>> labels = load_breast_cancer().target
    >>> x1 = X['mean_area'].values
    >>> x2 = X['mean_compactness'].values
    
    # Fit model with data.
    >>> model = impurity_binned_2d(bin_width='equal', 
    ...                            n_bins=20)
    >>> model.fit(x1, x2, labels)
    
    # Information table.
    >>> model.info
    
    # Output from each iteration.
    >>> model.bins_
    
    # Visualize the output.
    >>> model.plot(5)
    '''
    
    def __init__(self, n_bins=20, max_iter=10, tol=1e-4, 
                 bin_width='percentile'):
        
        self.n_bins = max(1, n_bins)
        self.max_iter = max(1, max_iter)
        self.tol = tol
        self.bin_width = bin_width
        self.eps = np.finfo(float).eps*10
    
    def __CheckArray__(self, x, name):
        
        '''Check array '''
        a = np.array(x)
        if not isinstance(x,(np.ndarray, list)):
            raise ValueError(f'`{name}` should be array-like or '
                             f'list, got {type(x)} instead.')
        elif isinstance(x, np.ndarray) & (len(a.shape)>1):
            raise ValueError(f'`{name}` should be 1D array, '
                             f'got {x.shape} instead.')
        elif a.dtype not in ('int','float'):
            raise ValueError(f'dtype of `{name}` should be ' 
                             f'numeric, got {a.dtype} instead.')
        elif np.isnan(a).sum()>0:
            raise ValueError(f'`{name}` should not contain NaN.')
        else: return x.copy()
    
    def __CheckBinedges__(self, x , bins, name):
        
        '''Check bin edges'''
        amin, amax = min(x), max(x)
        if bins is None:
            return [amin, amax+self.eps]
        else:
            a = self.__CheckArray__(bins, name)
            if min(np.diff(a))<=0:
                raise ValueError(f'`{name}` should be a sequence of '
                                 f'monotonically increasing floats, '
                                 f'got {a} instead.')
            elif (min(bins)>amin) | (max(bins)<=amax):
                raise ValueError(f'`{name}` should be [{amin},{amax}), ' 
                                 f'got [{min(bins)},{max(bins)}] instead.')
            return list(bins)
        
    def __CheckParams__(self, x1, x2):
        
        '''Check parameters'''
        a1 = self.__CheckArray__(x1, 'x1')                         
        a2 = self.__CheckArray__(x2, 'x2')                         
        
        if len(a1)!=len(a2):
            raise ValueError(f'`x1` and `x2` must have the same '
                             f'length, got (x1, x2)=({len(x1)}, '
                             f'{len(x2)}) instead.')
            
        if not isinstance(self.n_bins, int):
            raise ValueError(f'`n_bins` is a number of bins '
                             f'required. It should be > 0, got '
                             f'{self.n_bins} instead.')
            
        if self.max_iter <= 0:
            raise ValueError(f'`max_iter` should be > 0, got '
                             f'{self.max_iter} instead.')
            
        if not ((self.tol>=0) & (self.tol<1)):
            raise ValueError(f'`tol` should be [0,1), got '
                             f'{self.tol} instead.')
            
        if self.bin_width not in ("equal", "percentile"):
            raise ValueError(f'`bin_width` must be "equal", '
                             f'or "percentile", got '
                             f'{self.bin_width} instead.')
        return a1, a2
            
    def fit(self, x1, x2, labels, bin1=None, bin2=None):
        
        '''
        Fit model.
        
        Parameters
        ----------
        x1, x2 : array-like of shape (n_samples,)
        \t An array of scalars. 
        
        labels : array-like of shape (n_samples,)
        \t An array of class labels (`int`).
        
        bin1, bin2 : array-like, optional, default: None
        \t Initial bin edges. If None, it defaults to
        \t [min(`x`), max(`x`) + eps], where `eps` is the 
        \t next smallest representable float larger than 1.
        \t i.e. np.finfo(float).eps.
        
        Attributes
        ----------
        info : `pd.DataFrame` object
        \t Information table that contains output from  
        \t each iteration and is comprised of:
        \t - 'bin1'     : Number of `x1` bins 
        \t - 'bin2'     : Number of `x2` bins
        \t - 'gini'     : Gini impurity
        \t - 'add_bin1' : Added bin edge of `x1`
        \t - 'add_bin2' : Added bin edge of `x2`
        \t - 'pct_delta': Relative difference of Gini
        
        bins_ : `collections.OrderedDict` object
        \t An ordered dictionary that keeps the order of
        \t the keys. It contains named tuples, namely 
        \t "RandBins" with positions as follows:
        \t - 'combi'     : Number of bins of `x1`, and `x2`
        \t - 'gini'      : Weighted Gini impurity
        \t - 'bin_edges1': Bin edges of `x1`  
        \t - 'bin_edges2': Bin edges of `x2`
        \t - 'add_bin1'  : Added bin edge of `x1`
        \t - 'add_bin2'  : Added bin edge of `x2`
        '''
        # Initialize widget.
        t1 = widgets.HTMLMath(value='Initializing . . .')
        t2 = widgets.HTMLMath(value='')
        display(widgets.HBox([t1,t2]))
        time.sleep(1)
        update1 = lambda g: 'Calculating . . . Gini = {:.4f} '.format(g)
        update2 = lambda n: ('○  ' * (n%20), n+1)
     
        # Create product of 2 bin edges.
        a1, a2 = self.__CheckParams__(x1, x2)
        rng1 = self.__Binedges__(a1, self.n_bins, self.bin_width)
        rng2 = self.__Binedges__(a2, self.n_bins, self.bin_width)

        # Initialize parameters.
        n = len(np.unique(labels))
        self.init_gini = (1/n)*(1-1/n)*n
        Ginis = [self.init_gini]
        bin1 = self.__CheckBinedges__(a1 , bin1, 'bin1')
        bin2 = self.__CheckBinedges__(a2 , bin2, 'bin2')
        Ginis += [self.__2ArrayGini__(a1, a2, bin1, bin2, labels)]
        d = abs(float(np.diff(Ginis[-2:]))/Ginis[-2])
        deltas, r, i, n_iter = [d], 0, 0, 1
        
        # `rng` should not contain items similar to bin.
        rng1 = set(rng1).difference(bin1)
        rng2 = set(rng2).difference(bin2)
        has_items = min(len(rng1), len(rng2))>0
        
        # Create "collections" dictionary.
        info =  ['combi', 'gini', 'bin_edges1', 'bin_edges2', 
                 'add_bin1', 'add_bin2']
        Bins = collections.namedtuple('Bins', info)
        self.bins_ = collections.OrderedDict()
        self.bins_[0] = Bins(combi=[len(bin1)-1,len(bin2)-1], 
                             gini = Ginis[-1],
                             bin_edges1 = bin1, bin_edges2 = bin2,
                             add_bin1 = np.nan, add_bin2 = np.nan)
        
        while (deltas[-1]>self.tol) & has_items:
            
            min_gini = Ginis[-1]
            for p in product(rng1, rng2):
                
                r += 1
                if (r%100)==0: 
                    t1.value  = update1(min_gini)
                    t2.value, i  = update2(i)
                new_bin1, cond1 = self.__AddItem__(bin1, p[0])
                new_bin2, cond2 = self.__AddItem__(bin2, p[1])

                if not (cond1 & cond2):
                    args = (a1, a2, new_bin1, new_bin2, labels)
                    gini = self.__2ArrayGini__(*args)
                    if gini < min_gini:
                        fbins, min_gini = (new_bin1.copy(), 
                                           new_bin2.copy()), gini
            
            # Only record when Gini is improving.
            if min_gini!=Ginis[-1]:
                self.bins_[len(self.bins_)] = Bins(combi=[len(b)-1 for b in fbins], 
                                                   gini = min_gini,
                                                   bin_edges1 = fbins[0], 
                                                   bin_edges2 = fbins[1], 
                                                   add_bin1 = self.__diff__(fbins[0], bin1), 
                                                   add_bin2 = self.__diff__(fbins[1], bin2))
                
            # Update new bin edges.
            bin1, bin2 = fbins

            # Calculate `deltas`. If `delta` is either not 
            # improving or increasing, it will default to 0. 
            Ginis += [min_gini]
            d = float(np.diff(Ginis[-2:]))/Ginis[-2]
            deltas.append(abs(d) if d<0 else 0)
            
            # `rng` should not contain items similar to bin.
            rng1 = set(rng1).difference(bin1)
            rng2 = set(rng2).difference(bin2)
            has_items = min(len(rng1), len(rng2))>0
            
            # Stop when number of iterations exceeds `max_iter`.
            n_iter += 1
            if n_iter > self.max_iter: break
        
        # Store `x1`, `x2`, and `labels` in self.data.
        self.data = {'x1':a1,'x2':a2, 'labels':labels}
        
        # Create result dataframe (self.info).
        info = [[b.combi[0], b.combi[1], b.gini, b.add_bin1, 
                 b.add_bin2, deltas[i]] for i,b in self.bins_.items()]
        columns = ['bin1','bin2','gini','add_bin1','add_bin2','pct_delta']
        self.info = pd.DataFrame(info, columns=columns)
        t1.value, t2.value = 'Complete . . .', ''
        return self
                             
    def __diff__(self, a, b):
        
        '''The difference between two sets a and b '''
        d = list(set(a).difference(b))
        return d[0] if len(d)>0 else np.nan

    def __2ArrayGini__(self, a1, a2, bin1, bin2, labels):
        
        '''Calculate gini impurity'''
        # Group instances into bins.
        indx1 = np.digitize(a1, bin1)
        indx2 = np.digitize(a2, bin2)
        prods = np.unique(np.vstack((indx1,indx2)),
                          axis=1).T.tolist()
        k_bins,N = [labels[(indx1==i1) & (indx2==i2)] 
                    for i1,i2 in prods], a1.shape[0]

        # Determine gini impurity for each bin group.
        var = lambda x: np.mean(x)*(1-np.mean(x))
        gini = sum([var(k==n)*len(k)/N for k in k_bins 
                    for n in np.unique(k)])
        return gini

    def __AddItem__(self, a, i):
        
        '''Add item to list and then sort'''
        b = sorted(a + (i if isinstance(i,list) else [i]) 
                   if i not in a else a)
        return b, len(b)==len(a)

    def __Binedges__(self, a, bins=10, bin_width='equal'):
        
        '''Create bin edges'''
        q = np.arange(0,1,1/bins)
        if bin_width=='equal': 
            return q*(max(a)-min(a)+self.eps)+min(a)
        elif bin_width=='percentile':
            return np.unique(np.percentile(a, q*100))
        
    def plot(self, i=0, ax=None):

        '''
        Given nth-iteration (i), this function plots
        the corresponding output of bin edges i.e.
        `bin_edge1`, and `bin_edge2` on x, and y axis, 
        respectively. 
        
        Parameters
        ----------
        i : `int`, optional, default: None
        \t The ith iteration. If `i` exceeds number
        \t of iterations, it defaults to the maximum
        \t value i.e. len(self.info)-1.
        
        ax : `matplotlib.axes._subplots.AxesSubplot`
        \t The axis of the subplot. If not provided,  
        \t the axis is automatically created.
        
        Returns
        -------
        ax : `matplotlib.axes._subplots.AxesSubplot`
        \t Only when `ax` is provided.
        '''
        
        # Cap ith iteration.
        i = min(max(int(i),0),len(self.info)-1)
        has_ax = True
        
        if ax is None: 
            k = dict(figsize=(6.6,6))
            (fig, ax), has_ax = plt.subplots(**k), False
        
        # Scatter plot.
        labels = self.data['labels']
        for c in np.unique(labels):
            ax.scatter(self.data['x1'][labels==c],
                       self.data['x2'][labels==c], 
                       alpha=0.5, s=20, label=f'Label {c}', 
                       marker='s')
        
        # Find the nth-iteration of all add_bin2.
        a = self.info[['add_bin1']].sort_values(by='add_bin1')\
        .reset_index().values
        bins = self.bins_[i].bin_edges1[1:-1]
        niter = a[np.isin(a[:,1], bins),0].astype(int)
        
        # Plot `bin_edges1` on x-axis.
        y_min = ax.get_ylim()[1]
        for k,n in zip(niter, bins):
            ax.axvline(n, ls='--', lw=0.8, c='k')
            ax.annotate(s='{:,.3g} ({:,d})'.format(n,k), 
                        xy=(n, y_min), xytext=(0,10), 
                        ha='center', va='bottom',
                        textcoords='offset points', 
                        rotation=90)
        
        # Find the nth-iteration of all add_bin2.
        a = self.info[['add_bin2']].sort_values(by='add_bin2')\
        .reset_index().values
        bins = self.bins_[i].bin_edges2[1:-1]
        niter = a[np.isin(a[:,1], bins),0].astype(int)
        
        # Plot `bin_edges2` on y-axis.
        x_min = ax.get_xlim()[1]   
        for k,n in zip(niter, bins):
            ax.axhline(n, ls='--', lw=1, c='k')
            ax.annotate(s='{:,.3g} ({:,d})'.format(n,k), 
                        xy=(x_min,n), xytext=(10,0), 
                        ha='left', va='center', 
                        textcoords='offset points')

        # Text for computed Gini Impurtiy.
        gi, g0 = self.bins_[i].gini, self.init_gini
        ax.annotate(s = 'Gini = {:.3f} / {:.3g}'.format(gi,g0), 
                    xy = (ax.get_xlim()[0],ax.get_ylim()[0]), 
                    xytext=(5,5), textcoords='offset points',
                    ha='left', va='bottom',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='square', pad=0.3,
                              fc='white', ec='none'))
        
        # Show legends.
        ax.legend(loc=0)
        ax.set_xlabel('X1', fontsize=12, fontweight='bold')
        ax.set_ylabel('X2', fontsize=12, fontweight='bold')
        
        if has_ax==False:
            plt.tight_layout()
            plt.show()
