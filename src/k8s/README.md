# Kubernetes Initial Scripts for Project

This directory contains initial setup scripts for deploying and managing Kubernetes resources for the project. The script `initial_scripts_for_project.py` helps automate the creation of necessary YAML files and provides instructions for manual tasks.

## Contents

- `jenkins-deployment.yaml`: Deployment configuration for Jenkins.
- `api-gateway-hpa.yaml`: Horizontal Pod Autoscaler configuration for the API Gateway.

## Instructions

### Istio Installation

To install Istio, use the Istio CLI or Helm chart. For example, using Istioctl:

```sh
$ istioctl install --set profile=default -y
```
