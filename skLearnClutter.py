import pickle
from numpy import *
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import glob

fs=glob.glob('/media/grecu/ExtraDrive1/GVCode/GVCode/DPRSub/*')
n0=0
from scipy.ndimage import label, generate_binary_structure
zKuL=[]
ifig=0

for fname in fs:   # this loops through DPR files and extracts convective profiles over land
                   # satisfying some criteria
    fh=Dataset(fname[:],'r')
    zKu=fh['NS/PRE/zFactorMeasured'][:,:,:]
    zKa=fh['MS/PRE/zFactorMeasured'][:,:,:]
    h0=fh['NS/VER/heightZeroDeg'][:,22:26]
    flag=fh['NS/PRE/flagPrecip'][:,22:26]
    pType=(fh['NS/CSF/typePrecip'][:,22:26]/1e7).astype(int)
    clutF=fh['NS/PRE/binClutterFreeBottom'][:,22:26]
    hIce=fh['NS/CSF/flagHeavyIcePrecip'][:,22:26]
    a1=nonzero(flag>0)
    labeled_array, num_features = label(flag[:,2])
    print(num_features)
    cmax=0
    i1=0
    i2=flag.shape[0]
    for k in range(num_features):
        c1=nonzero(labeled_array==k+1)
        
        b1=nonzero(pType[c1[0],2]>0)
        if len(b1[0])>cmax:
            cmax=len(b1[0])
            i1=c1[0][0]-1
            i2=c1[0][-1]+1

    a2=nonzero(h0[a1]>4000)
    b2=nonzero(pType[a1][a2]==2)
    c2=nonzero(clutF[a1][a2][b2]>165)
    zKum=ma.array(zKu,mask=zKu<0)
    zKam=ma.array(zKa,mask=zKa<0)
    for j1,j2 in zip(a1[0][a2][b2][c2],a1[1][a2][b2][c2]):
        if hIce[j1,j2]>0:
            zKuL.append(zKum[j1,j2+22,:])
    n0+=len(c2[0])
    print(n0)
    print(h0.mean())
    if h0.mean()>4000 and cmax>5:
        plt.figure()
        plt.subplot(211)
        plt.pcolormesh(zKum[:,24,::-1].T,cmap='jet',vmax=45)
        plt.xlim(i1,i2)
        plt.ylim(0,120)
        plt.subplot(212)
        plt.pcolormesh(zKam[:,12,::-1].T,cmap='jet',vmax=35)#
        plt.xlim(i1,i2)
        plt.ylim(0,120)
        plt.savefig('crossSect%2.2i.png'%ifig)  # it also saves some plots
        plt.close('all')
    ifig+=1
    print(fname)
    plt.show()
    

zKuL=array(zKuL)
zKuL[zKuL<0]=0
import xarray as xr
zKux=xr.DataArray(zKuL)
d=xr.Dataset({'zKu':zKux})
d.to_netcdf('zKuDataBaseSample.nc')
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import neighbors
Xref=zKuL[:,0:165]  # this database is over land; binClutterFreeBottom>165
                    # only convective profiles with heavy ice and h0>4.00 km are included
                     
kmeans = KMeans(n_clusters=16, random_state=0).fit(Xref)  # kmeans clustering;
                                                          # it can be used for analysis/insight and prediction
                                                          # performance is poor for prediction
ic=0
for i in range(4):
    for j in range(4):
        ic+=1
        plt.plot(kmeans.cluster_centers_[ic-1],arange(165))
        plt.ylim(164,50)

y=Xref[:,-1]
X=Xref[:,100:150]  # only observations in bins 100:149 are used for prediction
nt=X.shape[0]      # the dataset is randomly split in two
r1=random.random(nt)
a=nonzero(r1<0.5)
b=nonzero(r1>0.5)
X_train=X[a[0],:]
X_valid=X[b[0],:]

y_train=y[a[0]]
y_valid=y[b[0]]


n_neighbors=16
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')   # kneighbors prediction
                                                                       # it does not work well
knn.fit(X_train,y_train)
yp=knn.predict(X_valid)
from sklearn.kernel_ridge import KernelRidge
krr = KernelRidge(kernel='poly', degree=1,alpha=5)                     # kernel ridge prediction
                                                                       # notable improvement relative to regression
                                                                       # of zKu at bin 149
krr.fit(X_train, y_train)
ypk=krr.predict(X_valid)
print(corrcoef(ypk,y_valid))
print((((ypk-y_valid)**2).mean())**0.5)
print((((X_valid[:,-1]-y_valid)**2).mean())**0.5)

