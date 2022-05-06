#!/bin/bash

# Train single networks
train_network -c simple.yml
train_network -c lstm.yml
train_network -c rnn.yml
predict -c predict.yml

# Train networks and vary the learning rate
train_networks -c simples_lr.yml
train_networks -c lstms_lr.yml

# Plot the different metrics
plot_metrics -c pproc_simples_lr.yml
plot_metrics -c pproc_lstms_lr.yml