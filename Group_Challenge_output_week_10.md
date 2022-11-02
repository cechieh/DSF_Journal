## Jupyter Notebook Exploratory Data Analysis Example
(Created 11-01-2022)

Save the Jupyter Notebook into your defined working directory.

Use this Docker [Jupyter Base Notebook](https://hub.docker.com/r/jupyter/base-notebook) to run this notebook.

Use: `docker pull jupyter/base-notebook`to download this Docker image to your computer.

Then execute: `docker run -it --rm -v "${PWD}":/home/jovyan/work -p 8888:8888 jupyter/base-notebook` 

The Docker container wil run in your default browser tab.

Your `$HOME` directory will be mapped into the `/work`directory inside your Jupyter Notebook environment.




```python

```


```python
# See if seaborn is installed
import seaborn as sns

sns.__version__
```




    '0.11.2'




```python
# Using the `pip` command install the latest Seaborn library and dependencies for this Jupyter Notebook
# Seaborn version greater than 0.12 is needed.
!pip install seaborn

# Load Seaborn into working memory
import seaborn as sns

# Prints the version of Seaborn
sns.__version__
```

    Requirement already satisfied: seaborn in c:\users\user\anaconda3\lib\site-packages (0.11.2)
    Requirement already satisfied: scipy>=1.0 in c:\users\user\anaconda3\lib\site-packages (from seaborn) (1.7.3)
    Requirement already satisfied: matplotlib>=2.2 in c:\users\user\anaconda3\lib\site-packages (from seaborn) (3.5.1)
    Requirement already satisfied: numpy>=1.15 in c:\users\user\anaconda3\lib\site-packages (from seaborn) (1.21.5)
    Requirement already satisfied: pandas>=0.23 in c:\users\user\anaconda3\lib\site-packages (from seaborn) (1.4.2)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\user\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (9.0.1)
    Requirement already satisfied: packaging>=20.0 in c:\users\user\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (21.3)
    Requirement already satisfied: cycler>=0.10 in c:\users\user\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (0.11.0)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\user\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (2.8.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (1.3.2)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (3.0.4)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\user\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (4.25.0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\user\anaconda3\lib\site-packages (from pandas>=0.23->seaborn) (2021.3)
    Requirement already satisfied: six>=1.5 in c:\users\user\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib>=2.2->seaborn) (1.16.0)
    




    '0.11.2'




```python
# Import necessary Python libraries
import numpy as np
import matplotlib.pyplot as plt

# Will use this Seaborn.objects
import seaborn.objects as so

import pandas as pd
#import sidetable

# SHIF+ENTER to execute cells
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Input In [30], in <cell line: 6>()
          3 import matplotlib.pyplot as plt
          5 # Will use this Seaborn.objects
    ----> 6 import seaborn.objects as so
          8 import pandas as pd
    

    ModuleNotFoundError: No module named 'seaborn.objects'



```python
# Seaborn includes sample datasets
sns.get_dataset_names()
```




    ['anagrams',
     'anscombe',
     'attention',
     'brain_networks',
     'car_crashes',
     'diamonds',
     'dots',
     'dowjones',
     'exercise',
     'flights',
     'fmri',
     'geyser',
     'glue',
     'healthexp',
     'iris',
     'mpg',
     'penguins',
     'planets',
     'seaice',
     'taxis',
     'tips',
     'titanic']




```python
# Load Penguins dataset
df0 = sns.load_dataset('titanic')
df0.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print dataframe information
df0.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   survived     891 non-null    int64   
     1   pclass       891 non-null    int64   
     2   sex          891 non-null    object  
     3   age          714 non-null    float64 
     4   sibsp        891 non-null    int64   
     5   parch        891 non-null    int64   
     6   fare         891 non-null    float64 
     7   embarked     889 non-null    object  
     8   class        891 non-null    category
     9   who          891 non-null    object  
     10  adult_male   891 non-null    bool    
     11  deck         203 non-null    category
     12  embark_town  889 non-null    object  
     13  alive        891 non-null    object  
     14  alone        891 non-null    bool    
    dtypes: bool(2), category(2), float64(2), int64(4), object(5)
    memory usage: 63.0+ KB
    


```python
# Print numer of missing values
df0.isnull().sum()
```




    survived         0
    pclass           0
    sex              0
    age            177
    sibsp            0
    parch            0
    fare             0
    embarked         2
    class            0
    who              0
    adult_male       0
    deck           688
    embark_town      2
    alive            0
    alone            0
    dtype: int64




```python
# Make a copy of original dataframe and create new working dataframe
df1 = df0.copy()

# Drop rwos with missing values
df1 = df1.dropna()

# Show dataframe information
df1.info()

```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 182 entries, 1 to 889
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   survived     182 non-null    int64   
     1   pclass       182 non-null    int64   
     2   sex          182 non-null    object  
     3   age          182 non-null    float64 
     4   sibsp        182 non-null    int64   
     5   parch        182 non-null    int64   
     6   fare         182 non-null    float64 
     7   embarked     182 non-null    object  
     8   class        182 non-null    category
     9   who          182 non-null    object  
     10  adult_male   182 non-null    bool    
     11  deck         182 non-null    category
     12  embark_town  182 non-null    object  
     13  alive        182 non-null    object  
     14  alone        182 non-null    bool    
    dtypes: bool(2), category(2), float64(2), int64(4), object(5)
    memory usage: 14.5+ KB
    


```python
# Descriptive Statistics of numeric variables
df1.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>182.000000</td>
      <td>182.000000</td>
      <td>182.000000</td>
      <td>182.000000</td>
      <td>182.000000</td>
      <td>182.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.675824</td>
      <td>1.192308</td>
      <td>35.623187</td>
      <td>0.467033</td>
      <td>0.478022</td>
      <td>78.919735</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.469357</td>
      <td>0.516411</td>
      <td>15.671615</td>
      <td>0.645007</td>
      <td>0.755869</td>
      <td>76.490774</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.920000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>24.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>29.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>36.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>57.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>47.750000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Descriptive statistocs of categorical variables
df1[ ['sibsp', 'parch', 'pclass']].describe() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sibsp</th>
      <th>parch</th>
      <th>pclass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>182.000000</td>
      <td>182.000000</td>
      <td>182.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.467033</td>
      <td>0.478022</td>
      <td>1.192308</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.645007</td>
      <td>0.755869</td>
      <td>0.516411</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Data visualization with Seaborn

**References**
- [Seaborn documentation](https://seaborn.pydata.org/tutorial/introduction.html)
- [Seaborn.objects documentation](https://seaborn.pydata.org/generated/seaborn.objects.Plot.html#)
- [Data Visualization with Python](https://github.com/clizarraga-UAD7/Workshops/wiki/Data-Visualization-with-Python)
- [Data Visualization with Python - Jupyter Notebook Example](https://github.com/clizarraga-UAD7/Notebooks/blob/main/Data_Visualization_with_Python2.ipynb)



```python
# Redefine working dataframe to df
df = df1.copy()
```


```python
# Seaborn 

#define figure size and Seaborn style
sns.set(rc={"figure.figsize":(6, 4)}) #width=12, height=8
sns.set_style("whitegrid")

# Scatterplot
sns.scatterplot(data=df, x="age", y="fare", hue="sex", style="survived").set(
    xlabel="age", ylabel="fare", title="Seaborn Scatter Plot");


```


    
![png](output_14_0.png)
    



```python
# With Seaborn objects
from seaborn import axes_style
    
sns.scatterplot(data=df, x="sibsp", y="parch", color="pclass", marker="sex")
    .add(so.Dot()).layout(size=(6, 4)).theme({**axes_style("whitegrid"), "grid.linestyle": ":"})
    .label(x="sibsp", y="parch", color="pclass",title="Seaborn objects scatterplot")
)
```


      Input In [24]
        .add(so.Dot()).layout(size=(6, 4)).theme({**axes_style("whitegrid"), "grid.linestyle": ":"})
        ^
    IndentationError: unexpected indent
    



```python

# Using Seaborn barplot

sns.barplot(x=df['age'], title="Seaborn Bar Plot");


```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Input In [28], in <cell line: 3>()
          1 # Using Seaborn barplot
    ----> 3 sns.barplot(x=df['age'], title="Seaborn Bar Plot")
    

    File ~\anaconda3\lib\site-packages\seaborn\_decorators.py:46, in _deprecate_positional_args.<locals>.inner_f(*args, **kwargs)
         36     warnings.warn(
         37         "Pass the following variable{} as {}keyword arg{}: {}. "
         38         "From version 0.12, the only valid positional argument "
       (...)
         43         FutureWarning
         44     )
         45 kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 46 return f(**kwargs)
    

    File ~\anaconda3\lib\site-packages\seaborn\categorical.py:3190, in barplot(x, y, hue, data, order, hue_order, estimator, ci, n_boot, units, seed, orient, color, palette, saturation, errcolor, errwidth, capsize, dodge, ax, **kwargs)
       3187 if ax is None:
       3188     ax = plt.gca()
    -> 3190 plotter.plot(ax, kwargs)
       3191 return ax
    

    File ~\anaconda3\lib\site-packages\seaborn\categorical.py:1639, in _BarPlotter.plot(self, ax, bar_kws)
       1637 def plot(self, ax, bar_kws):
       1638     """Make the plot."""
    -> 1639     self.draw_bars(ax, bar_kws)
       1640     self.annotate_axes(ax)
       1641     if self.orient == "h":
    

    File ~\anaconda3\lib\site-packages\seaborn\categorical.py:1604, in _BarPlotter.draw_bars(self, ax, kws)
       1599 barpos = np.arange(len(self.statistic))
       1601 if self.plot_hues is None:
       1602 
       1603     # Draw the bars
    -> 1604     barfunc(barpos, self.statistic, self.width,
       1605             color=self.colors, align="center", **kws)
       1607     # Draw the confidence intervals
       1608     errcolors = [self.errcolor] * len(barpos)
    

    File ~\anaconda3\lib\site-packages\matplotlib\axes\_axes.py:2551, in Axes.barh(self, y, width, height, left, align, **kwargs)
       2452 r"""
       2453 Make a horizontal bar plot.
       2454 
       (...)
       2548 .
       2549 """
       2550 kwargs.setdefault('orientation', 'horizontal')
    -> 2551 patches = self.bar(x=left, height=height, width=width, bottom=y,
       2552                    align=align, **kwargs)
       2553 return patches
    

    File ~\anaconda3\lib\site-packages\matplotlib\__init__.py:1412, in _preprocess_data.<locals>.inner(ax, data, *args, **kwargs)
       1409 @functools.wraps(func)
       1410 def inner(ax, *args, data=None, **kwargs):
       1411     if data is None:
    -> 1412         return func(ax, *map(sanitize_sequence, args), **kwargs)
       1414     bound = new_sig.bind(ax, *args, **kwargs)
       1415     auto_label = (bound.arguments.get(label_namer)
       1416                   or bound.kwargs.get(label_namer))
    

    File ~\anaconda3\lib\site-packages\matplotlib\axes\_axes.py:2403, in Axes.bar(self, x, height, width, bottom, align, **kwargs)
       2394 for l, b, w, h, c, e, lw, htch in args:
       2395     r = mpatches.Rectangle(
       2396         xy=(l, b), width=w, height=h,
       2397         facecolor=c,
       (...)
       2401         hatch=htch,
       2402         )
    -> 2403     r.update(kwargs)
       2404     r.get_path()._interpolation_steps = 100
       2405     if orientation == 'vertical':
    

    File ~\anaconda3\lib\site-packages\matplotlib\artist.py:1064, in Artist.update(self, props)
       1062             func = getattr(self, f"set_{k}", None)
       1063             if not callable(func):
    -> 1064                 raise AttributeError(f"{type(self).__name__!r} object "
       1065                                      f"has no property {k!r}")
       1066             ret.append(func(v))
       1067 if ret:
    

    AttributeError: 'Rectangle' object has no property 'title'



    
![png](output_16_1.png)
    



```python
# Seaborn objects barplot
(
    so.Plot(df, x="age", y="plass", color="gender")
    .add(so.Bar(), so.Agg())
    .label(
        x="age", y="pclass",title="Seaborn objects scatterplot"))

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [27], in <cell line: 3>()
          1 # Seaborn objects barplot
          2 (
    ----> 3     so.Plot(df, x="age", y="plass", color="gender")
          4     .add(so.Bar(), so.Agg())
          5     .label(
          6         x="age", y="pclass",title="Seaborn objects scatterplot"))
    

    NameError: name 'so' is not defined



```python
# Using Seaborn barplot

sns.barplot(data=df, x='species',y='flipper_length_mm', hue='sex', dodge=True).set(ylabel="Flipper length (mm)", 
                                                                                   title="Seaborn Dodged Bar Plot");

```


    
![png](output_18_0.png)
    



```python
# Seaborn objects barplot
(
    so.Plot(df, x="species", y="flipper_length_mm", color="sex")
    .add(so.Bar(), so.Agg(), so.Dodge())
    .label(
        x="Species", y="Flipper depth (mm)",title="Seaborn objects dodged scatterplot"
    )
)
```




    
![png](output_19_0.png)
    




```python
# Seaborn histogram

sns.histplot(df['body_mass_g']).set(
    xlabel="Body mass (g)", title="Seaborn Histogram Plot");

```


    
![png](output_20_0.png)
    



```python
# Seaborn objects histogram

(
    so.Plot(df, x="age")
    .add(so.Bars(), so.Hist()).label(
        x="age",title="Seaborn objects histogram"))

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [29], in <cell line: 4>()
          1 # Seaborn objects histogram
          3 (
    ----> 4     so.Plot(df, x="age")
          5     .add(so.Bars(), so.Hist()).label(
          6         x="age",title="Seaborn objects histogram"))
    

    NameError: name 'so' is not defined



```python
# Define basic plot and add properties
p = so.Plot(df, "body_mass_g", color="species")
p.add(so.Bars(), so.Hist()).label(
        x="Body mass(g)",title="Seaborn objects histogram (non-stacked)"
    )
```




    
![png](output_22_0.png)
    




```python
p.add(so.Bars(), so.Hist(), so.Stack(), color="species").label(
        x="Body mass(g)",title="Seaborn objects histogram (stacked)"
    )
```




    
![png](output_23_0.png)
    




```python
sns.set(rc={"figure.figsize":(4, 4)})
g = sns.FacetGrid(df, col="species",  row="sex")
g.map_dataframe(sns.histplot, x="body_mass_g");

```


    
![png](output_24_0.png)
    



```python
(
    so.Plot(df, x="body_mass_g")
    .facet(col="species", row="sex")
    .add(so.Bars(), so.Hist()).label(
        x="Body mass (g)"
    ).layout(size=(10, 6))
)

```




    
![png](output_25_0.png)
    



## Not yet implemented in seaborn.objects


```python
# Seaborn boxplots

sns.boxplot(data=df, x="species", y="flipper_length_mm").set(
    ylabel="Flipper length (mm)", title="Seaborn Boxplot Plot");

```


    
![png](output_27_0.png)
    



```python
sns.catplot(data=df, x='species',y='flipper_length_mm', kind='box', col='island');
```


    
![png](output_28_0.png)
    



```python
# Seaborn violin plot

sns.violinplot(data=df, x='species', y='body_mass_g', hue='island', height=12);
```


    
![png](output_29_0.png)
    



```python
# Seaborn violin plot

sns.violinplot(data=df, x='species', y='body_mass_g', hue='sex',
               split=True, inner="quart", linewidth=1);
```


    
![png](output_30_0.png)
    



```python
# Seaborn Masked correlation heatmap 
#corr = df.corr()
# Select only numeric columns
corr = df.select_dtypes(include=np.number).corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
  f, ax = plt.subplots(figsize=(8, 8))
  ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True, cmap='Blues');

```


    
![png](output_31_0.png)
    



```python
# Seaborn KDE plots

sns.FacetGrid(df, hue="species", height=6,) \
   .map(sns.kdeplot, "flipper_length_mm", fill=True) \
   .add_legend();

```


    
![png](output_32_0.png)
    



```python
# Seaborn pairplot using KDE plots
sns.pairplot(df, hue='species', height=3);

```


    
![png](output_33_0.png)
    



```python
# Pairplot using histograms
sns.pairplot(df, hue="species", height=3, diag_kind="hist");
```


    
![png](output_34_0.png)
    



```python

```


```python

```


```python

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Input In [2], in <cell line: 6>()
          3 import matplotlib.pyplot as plt
          5 # Will use this Seaborn.objects
    ----> 6 import seaborn.objects as so
          8 import pandas as pd
    

    ModuleNotFoundError: No module named 'seaborn.objects'



```python

```




    '0.11.2'




```python

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Input In [4], in <cell line: 6>()
          3 import matplotlib.pyplot as plt
          5 # Will use this Seaborn.objects
    ----> 6 import seaborn.objects as so
          8 import pandas as pd
    

    ModuleNotFoundError: No module named 'seaborn.objects'



```python

```
