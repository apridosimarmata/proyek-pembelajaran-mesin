# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import sys
LE = LabelEncoder()

args = sys.argv

url = 'https://raw.githubusercontent.com/apridosimarmata/proyek-pembelajaran-mesin/main/cafe_data.csv'
data = pd.read_csv(f'uploads/{args[1]}')
data.head()

data['Code'] = LE.fit_transform(data['Category'])
data['Date'] = data['Date'].str.split(' ')
data['Date'] = [ int(l[1].split(':')[0]) for l in data['Date']]
data = data.rename(columns={'Date': 'Hour'})
data['Code'] = LE.fit_transform(data['Category'])

cols = list(data.columns.values)
cols.pop(cols.index('Hour'))
data = data[cols+['Hour']]

cols = list(data.columns.values)
cols.pop(cols.index('Rate'))
data = data[cols+['Rate']]

cols = list(data.columns.values)
cols.pop(cols.index('Code'))
data = data[cols+['Code']]
data.head()

data['Hour'].min()
data['Hour'].max()
X = data.iloc[:,1:4].values

#elbow method
wcss = []
for i in range(1,11):
    k_means = KMeans(n_clusters=i,init='k-means++', random_state=42)
    k_means.fit(X)
    wcss.append(k_means.inertia_)

max_change = 0

max_idx = 0

for idx in range(0, len(wcss) - 1):
    change = abs(wcss[idx+1] - wcss[idx])
    if change > max_change:
        max_change = change
        max_idx = idx + 2

k_means_optimum = KMeans(n_clusters = max_idx, init = 'k-means++',  random_state=42)
y = k_means_optimum.fit_predict(X)

data['cluster'] = y
clusters = {}
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for cluster in range(0, max_idx):
    clusters[cluster] = data[data.cluster==cluster]

kplot = plt.axes(projection='3d')
xline = np.linspace(0, 15, 1000)
yline = np.linspace(0, 15, 1000)
zline = np.linspace(0, 15, 1000)
kplot.plot3D(xline, yline, zline, 'black')
# Data for three-dimensional scattered points

for cluster in clusters.keys():
    kplot.scatter3D(clusters[cluster].Hour, clusters[cluster].Rate, clusters[cluster].Code, c = colors[cluster], label = f'Cluster {cluster}')    

plt.scatter(k_means_optimum.cluster_centers_[:,0], k_means_optimum.cluster_centers_[:,1], color = 'indigo', s = 200)
plt.legend()

plt.title("Kmeans")

plt.savefig(f'{args[1]}.png')

import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders


def send_mail(send_from, send_to, subject, message, files=[],
              server="smtp.gmail.com", port=587, username='ifs18034@gmail.com', password='ponsel11',
              use_tls=True):
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = send_to
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message))

    for path in files:
        part = MIMEBase('application', "octet-stream")
        with open(path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename={}'.format(Path(path).name))
        msg.attach(part)

    smtp = smtplib.SMTP(server, port)
    smtp.set_debuglevel(1)
    if use_tls:
        smtp.starttls()
    smtp.login(username, password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()

send_mail("Proyek PM", args[2], "Hasil Clustering Anda", "Berikut terlampir hasil clustering data penjualan toko Anda.", [f'{args[1]}.png'])
