# GRANO-GAN

## Overview
This is an implemetation on GRANO-GAN algorithm which performs anomaly detection in time series data. The details of this can be found in paper "GRANO-GAN: Gradient Normalized Anomaly Detection GAN"

GRANO-GAN belongs to the category of unsupervised learning models. The architecture comprosies of most latest advancements made in GAN's leading to better score.

This algorithm has been evaluated on several datasets.
<ul>
<li>NAB</li>
    The NAB corpus of 58 timeseries data files is designed to provide data for research in streaming anomaly detection. It is comprised of both real-world and artifical timeseries data containing labeled anomalous periods of behavior. Data are ordered, timestamped, single-valued metrics. All data files contain anomalies, unless otherwise noted.
<li>Yahoo</li>
    The goal of this dataset is to benchmark your anomaly detection algorithm. The dataset consists of real and synthetic time-series with tagged anomaly points. The dataset tests the detection accuracy of various anomaly-types including outliers and change-points. The synthetic dataset consists of time-series with varying trend, noise and seasonality. The real dataset consists of time-series representing the metrics of various Yahoo services

<li>Swat</li>
    Network traffic & all the values obtained from all the 51 sensors and actuators . Data labelled according to normal and abnormal behaviours
<li>Wadi</li>
    16 days of continuous operation: 14 under normal operation & 2 days with attack scenarios. This contains data from all the 123 sensors and actuators
</ul>

The new algorithm has proven to be effective when distribution is complex and can learn it well. There is still room for future scope in performance improvement and fine tuning of layers in the algorithm.

## Authors/Contributors
* [Naveen Rudrappa](https://github.com/NaveenRudra)
* [Somedip Karmakar](https://www.linkedin.com/in/somedip-karmakar-1b180742/?originalSubdomain=in)
* [Shashi Jain](https://www.linkedin.com/in/drshashijain/?originalSubdomain=nl)