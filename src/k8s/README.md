# Kubernetes Initial Scripts for Generating a Project by Python

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
Enable Istio injection for the default namespace:


```sh
$ kubectl label namespace default istio-injection=enabled
```

### TLS/SSL Configuration

To configure TLS/SSL for all services, ensure your Ingress or Gateway resources are configured with TLS certificates. You may use cert-manager for automatic certificate provisioning:

```sh
kubectl create namespace cert-manager
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager --namespace cert-manager --create-namespace --set installCRDs=true
```

## Usage

Run the script to generate the YAML files for Jenkins deployment and API Gateway Horizontal Pod Autoscaler:

```sh
python initial_scripts_for_project.py
```
This will generate the following files:

 - `jenkins-deployment.yaml`
 - `api-gateway-hpa.yaml`

These files can then be applied to your Kubernetes cluster using `kubectl apply -f <file>.`

```sh
kubectl apply -f jenkins-deployment.yaml
kubectl apply -f api-gateway-hpa.yaml
```

## Summary
The script provides a starting point for setting up CI/CD pipelines and managing scalability in your Kubernetes cluster. Follow the instructions for manual tasks and use the generated YAML files to automate deployments.

## Additional Resources

For more information on the tools and concepts mentioned in this README, refer to the following resources:

- [Istio Documentation](https://istio.io/latest/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [Helm Documentation](https://helm.sh/docs/)
- [cert-manager Documentation](https://cert-manager.io/docs/)
- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [Kubernetes Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [YAML Language Reference](https://yaml.org/spec/1.2/spec.html)
