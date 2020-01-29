import math

import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from matplotlib import pyplot as plt
from sklearn import preprocessing as pre
from sklearn.cluster import KMeans


def radius(f, l, r):
    """

    :param f:
    :param l:
    :param r:
    :return:
    """
    df = r / 69  # North-south distance in degrees
    dl = df / math.cos(f)  # East-west distance in degrees
    # x1,x2,y1,y2=0.0 #r means radius in miles
    x1 = (f - df)
    # x1=round(x1,6)
    x2 = (f + df)
    y1 = (l - dl)
    y2 = (l + dl)
    coordinates = [y2, y1, x1, x2]
    return coordinates
    # {(f-df,l-dl), (f+df,l-dl), (f+df,l+dl), (f-df,l+dl)} #List of vertices
    # '''df = 10/69 = 0.145
    # dl = 0.145 / cos(50 degrees) = 0.145 / 0.6428 = 0.225
    # f - df = 50 - 0.145 = 49.855 #(southernmost latitude)
    # f + df = 50 + 0.145 = 50.145 #(northernmost latitude)
    # l - dl = -1 - 0.225 = -1.225 #(western longitude)
    # l + dl = -1 + 0.225 = -0.775 #(eastern longitude)'''


def plot1(xyz):
    fig = plt.figure()
    kmeans = KMeans(n_clusters=10, init='k-means++')
    kmeans.fit(xyz[xyz.columns[1:3]])  # Compute k-means clustering.
    xyz['cluster_label'] = kmeans.fit_predict(xyz[xyz.columns[1:3]])
    centers = kmeans.cluster_centers_  # Coordinates of cluster centers.
    labels = kmeans.predict(xyz[xyz.columns[1:3]])  # Labels of each point
    xyz.head(10)
    xyz.plot.scatter(x='Latitude', y='Longitude', c=labels, s=50, cmap='viridis')
    tt = plt.scatter(centers[:, 0], centers[:, 1], c='Black', s=200, alpha=0.5)
    # print(tt)
    fig = plt.figure()
    fig.savefig('plot.png')


def rmoutlier(y, Const):  # function to remove outliers of locations
    # -87.9977,-87.5336,41.5600,42.1860
    y.drop(y[y['Longitude'] <= Const[0]].index, inplace=True, axis=0)
    y.drop(y[y['Longitude'] >= Const[1]].index, inplace=True, axis=0)
    y.drop(y[y['Latitude'] <= Const[2]].index, inplace=True, axis=0)
    y.drop(y[y['Latitude'] >= Const[3]].index, inplace=True, axis=0)
    return y


def elbowCurve(result1):
    K_clusters = range(1, 10)
    kmeans = [KMeans(n_clusters=i) for i in K_clusters]
    Y_axis = result1[['Latitude']]
    X_axis = result1[['Longitude']]
    score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]  # Visualize
    plt.plot(K_clusters, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()


def geo_code(geo):
    locator = Nominatim(user_agent='myGeocoder', timeout=3)
    # 1 - convenient function to delay between geocode calls
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
    # 2- - create location column
    locationEntered = locator.geocode(geo)
    # 3 - create longitude, latitude and altitude from location column (returns tuple)
    # geo['point'] = geo['location'].apply(lambda loc: tuple(loc.point) if loc else None)
    # 4 - split point column into latitude, longitude and altitude columns
    # geo[['latitude', 'longitude', 'altitude']] = pd.DataFrame(geo['point'].tolist(),index=geo.index)
    return locationEntered


def SquareRegion(ab, x):  # applying region coordinates to map
    """

    :rtype: object 
    """
    xregion = x.loc[:, ['ID', 'Latitude', 'Longitude', 'Arrest']]
    # xregion.head(10)
    xregion = rmoutlier(xregion, ab)
    return xregion


class dmain():
    crime = pd.read_csv('chicago2019.csv')
    crime1 = pd.read_csv('chicago2018.csv')
    result1 = pd.concat([crime, crime1], axis=0)  # to merge two csv files
    result1.dropna(inplace=True)  # removing null values
    label_encoder = pre.LabelEncoder()
    result1['Arrest'] = label_encoder.fit_transform(result1['Arrest'])
    result1['Domestic'] = label_encoder.fit_transform(result1['Domestic'])
    BBox = (result1.Longitude.min(), result1.Longitude.max(), result1.Latitude.min(), result1.Latitude.max())
    x = result1.loc[:, ['ID', 'Latitude', 'Longitude', 'Arrest']]

    # elbowCurve(result1)          #call this fun if needed to visualize elbow curve

    mapConstraints = [-87.9977, -87.5336, 41.5600, 42.1860]  # standard location constraints for chicago city
    x = rmoutlier(x, mapConstraints)
    z = result1.loc[:, ['ID', 'Latitude', 'Longitude']]

    # plot1(z)  # plotting graph before removing null values
    # plot1(x)  # plotting after removing null values
    # x region out line  box having values
    xBBox = (x.Longitude.min(), x.Longitude.max(), x.Latitude.min(), x.Latitude.max())

    # giving location coordinates manually
    ab = radius(41.853418, -87.730543, 2)

    # made it dynamic here

    xregion = SquareRegion(ab, x)
    # ploting xregion i.e reduced region along given radious
    plot1(xregion)
    arrper = (xregion['Arrest'].sum() / xregion['Arrest'].count()) * 100
    # arr_prr means arrest percentage in a given region
    print(arrper)
    # locationEntered=('Champ de Mars, Paris, France')
    locationEntered = '1645,West 47th Street,Chicago'
    # calling for geo codes
    locationEntered = geo_code(locationEntered)
    # calling the radius function for given address block
    ac = radius(locationEntered.latitude, locationEntered.longitude, 2)
    xregion = SquareRegion(ac, x)
    arrper = (xregion['Arrest'].sum() / xregion['Arrest'].count()) * 100
    # arr_prr means arrest percentage in a given region
    print(arrper)
    if arrper > 75.0:
        print('green-safe')
    if 50.0 <= arrper <= 75.0:
        print('orange-above average')
    if arrper < 50.0:
        print('red -less than 50 percent cases are solved')
