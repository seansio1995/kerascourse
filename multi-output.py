import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import SGD
from keras.preprocessing import text
import numpy as np
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

############################################################################

data = pd.read_csv("data/Video-Games-Sales-as-at-22-Dec-2016.csv")
print(data.values.shape)

data = data.loc[data["NA_Sales"]>1]
data = data.loc[data["EU_Sales"]>1]

data = data.dropna(axis=0)
print(data.values.shape)
names = data[["Name","Year_of_Release"]]

NA = data["NA_Sales"].values
EU = data["EU_Sales"].values
JP = data["JP_Sales"].values
X  = data[["Critic_Score","Critic_Count","User_Count","Year_of_Release"]].values

platform  = pd.get_dummies(data["Platform"]).values
genre     = pd.get_dummies(data["Genre"]).values
publisher = pd.get_dummies(data["Publisher"]).values
rating    = pd.get_dummies(data["Rating"]).values


X = np.concatenate((X,platform,genre,publisher,rating),axis=1)
############################################################################

a = Input(shape=(60,))
c = Dense(32, activation='sigmoid')(a)
d = Dense(32, activation='sigmoid')(c)
fina1 = Dense(1,activation="linear")(d)
fina2 = Dense(1,activation="linear")(d)
fina3 = Dense(1,activation="linear")(d)
model = Model(input=a, output=[fina1,fina2,fina3])


model.compile(optimizer='adam', loss='mse')
model.fit(X, [NA,EU,JP] ,nb_epoch=3000, batch_size=100, verbose=2)
p_NA,p_EU,p_JP = model.predict(X)


predictions = pd.DataFrame(np.concatenate((names,p_NA,NA[:, np.newaxis],p_EU,EU[:, np.newaxis],p_JP,JP[:, np.newaxis]),axis=1))
predictions.columns = (["Name","Year","p_NA","NA","p_EU","EU","p_JP","JP"])


print(predictions)

############################################################################
############################################################################
'''
Epoch 3000/3000
 - 0s - loss: 7.3309 - dense_3_loss: 4.3671 - dense_4_loss: 2.1512 - dense_5_loss: 0.8126
                                             Name  Year     p_NA     NA     p_EU     EU       p_JP    JP
0                                      Wii Sports  2006  23.6885  41.36  18.2142  28.96    5.11542  3.77
1                                  Mario Kart Wii  2008    18.48  15.68  13.7546  12.76    4.63455  3.79
2                               Wii Sports Resort  2009  12.7249  15.61  9.68986  10.93    2.81554  3.28
3                           New Super Mario Bros.  2006  8.05638  11.28  5.64991   9.14    2.31112   6.5
4                                        Wii Play  2006  3.55355  13.96  2.68191   9.18   0.736391  2.93
5                       New Super Mario Bros. Wii  2009  14.1234  14.44  10.3203   6.94    3.74166   4.7
6                                   Mario Kart DS  2005  9.21202   9.71  6.52806   7.47    2.60256  4.13
7                                         Wii Fit  2007  10.1431   8.92  7.70447   8.03    2.24984   3.6
8                              Kinect Adventures!  2010  2.47267     15  1.86771   4.89   0.473382  0.24
9                                    Wii Fit Plus  2009  7.19996   9.01  5.45051   8.49    1.59017  2.53
10                             Grand Theft Auto V  2013  4.23405   7.02  2.87281   9.09   0.268845  0.98
11   Brain Age: Train Your Brain in Minutes a Day  2005  2.80003   4.74  2.11394    9.2   0.553596  4.16
12                             Grand Theft Auto V  2013  4.45841   9.66  3.00373   5.14   0.284739  0.06
13                    Grand Theft Auto: Vice City  2002  6.23598   8.41  4.28253   5.49    1.83245  0.47
14    Brain Age 2: More Training in Minutes a Day  2005  2.49031   3.43  1.88097   5.35   0.477721  5.32
15                         Gran Turismo 3: A-Spec  2001  2.52539   6.85  1.90683   5.09    0.48686  1.87
16                 Call of Duty: Modern Warfare 3  2011  3.15787   9.04   2.6255   4.24  0.0621691  0.13
17                        Call of Duty: Black Ops  2010  4.55845    9.7  2.85015   3.68    0.45261  0.11
18                     Call of Duty: Black Ops II  2012  3.59274   4.99  2.55492   5.73   0.567828  0.65
19                     Call of Duty: Black Ops II  2012  4.59078   8.25  2.86927   4.24   0.454739  0.07
20                 Call of Duty: Modern Warfare 2  2009  4.47886   8.52  3.01572   3.59   0.286188  0.08
21                 Call of Duty: Modern Warfare 3  2011  3.71648   5.54  2.58562   5.73  0.0926611  0.49
22                           Grand Theft Auto III  2001  3.94428   6.99   2.5897   4.51     1.2017   0.3
23                        Super Smash Bros. Brawl  2008  2.99903   6.62   1.9645   2.55    0.35221  2.66
24                                   Mario Kart 7  2011  3.66052   5.03  2.38321   4.02    1.12134  2.69
25                        Call of Duty: Black Ops  2010  3.75473   5.99  2.66886   4.37   0.595013  0.48
26                             Grand Theft Auto V  2014  2.86102   3.96  2.10127   6.31   0.177319  0.38
27                    Animal Crossing: Wild World  2005  2.78464    2.5  2.10234   3.45    0.54985  5.33
28                                         Halo 3  2007  4.47888   7.97  3.01573   2.81   0.286189  0.13
29                             Super Mario Galaxy  2007  4.40073   6.06   2.7574   3.35   0.442295   1.2
..                                            ...   ...      ...    ...      ...    ...        ...   ...
228                            Fallout: New Vegas  2010  1.98437   1.52  1.18025   1.03   0.640512   0.1
229    The Legend of Zelda: A Link Between Worlds  2013  3.40264    1.4  2.41352   1.01   0.529914  0.46
230                            The Sims: Vacation  2002  2.46761   1.72  1.86391   1.21   0.472142     0
231                    LEGO Batman: The Videogame  2008  2.46762   1.75  1.86392   1.01   0.472145     0
232                                    Heavy Rain  2010  3.36408   1.29  2.37716   1.21   0.208755  0.06
233                    Tom Clancy's Splinter Cell  2002  2.46763   1.85  1.86392   1.04   0.472147     0
234                                    Killzone 2  2009  3.00481    1.4  1.96752   1.06   0.352486  0.08
235                                     Far Cry 3  2012  3.77023   1.38  2.68653   1.32   0.602726  0.02
236                         The Sims: Livin Large  2000  2.46762   1.67  1.86392   1.18   0.472146     0
237                   Star Wars: The Old Republic  2011   2.8597   1.58  2.10055   1.01   0.177243     0
238                Middle-Earth: Shadow of Mordor  2014   2.9985   1.01  1.96421   1.43   0.352173  0.05
239                                  Sonic Heroes  2003  2.46962   1.04  1.86542   1.37   0.472635  0.06
240                     Hitman 2: Silent Assassin  2002  2.47062   1.36  1.86617   1.15   0.472883  0.04
241                    Battlefield: Bad Company 2  2010  2.02074   1.33  1.20599    1.1   0.651005  0.08
242                    Sonic Mega Collection Plus  2004    2.473   1.54  1.86796   1.14   0.473466     0
243                        Cooking Mama: Cook Off  2007  2.46816   1.41  1.86433   1.12   0.472279  0.05
244                     Call of Duty: Finest Hour  2004  2.46892   1.51  1.86489   1.12   0.472465  0.01
245               Assassin's Creed IV: Black Flag  2013  2.99707   1.07  1.96316   1.31   0.351914  0.06
246                                Medal of Honor  2010  2.34024   1.28  1.72772   1.04   0.480693  0.07
247                   Need for Speed: Hot Pursuit  2010  1.97019   1.05  1.21103   1.23   0.600504  0.03
248                 Skylanders: Spyro's Adventure  2011  2.46762   1.35  1.86392   1.13   0.472146     0
249                                      The Sims  2003  2.46762   1.41  1.86391   1.12   0.472144     0
250                         Saints Row: The Third  2011   1.9721   1.25  1.17156   1.14   0.636967  0.07
251                    Sonic and the Secret Rings  2007  2.47142   1.21  1.86677   1.19   0.473074  0.04
252                           Burnout 3: Takedown  2004  2.46778   1.23  1.86404   1.11   0.472188     0
253                                Medal of Honor  1998  2.46768   1.44  1.86397   1.09   0.472161     0
254                    Tom Clancy's Splinter Cell  2003  2.46768   1.15  1.86396   1.11   0.472159     0
255                        Mario Strikers Charged  2007  2.54926   1.05  1.92529   1.05   0.492197  0.24
256                                    Crazy Taxi  2001  2.46764   1.13  1.86393   1.12   0.472151  0.06
257                         The Sims: Bustin' Out  2003  2.46762   1.07  1.86391   1.19   0.472144     0

[258 rows x 8 columns]
'''
