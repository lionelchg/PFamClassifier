#!/bin/bash

# Train single networks
train_network -c simple_5000labels.yml
train_network -c simple_15000labels.yml
predict -c predict.yml

plot_metrics -c pproc.yml