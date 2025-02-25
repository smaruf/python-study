# Kubernetes Initial Scripts for Generating a Project by Python

The `k8s` folder contains scripts and configurations for deploying and managing Kubernetes resources for the project. Below is an overview of the contents and their purposes:

#### Contents

- `jenkins-deployment.yaml`: Deployment configuration for Jenkins.
- `api-gateway-hpa.yaml`: Horizontal Pod Autoscaler configuration for the API Gateway.
- `setup_k8s.py`: Script for setting up Kubernetes resources including Jenkins, Istio, and Horizontal Pod Autoscaler.
- `k8s_management.py`: Script for managing Kubernetes resources and configurations, including monitoring and predictive scaling.
- `springboot_generator.py`: Script for generating Spring Boot project structure with Kubernetes configurations.
- `terraform/py_generate_tr.py`: Script for generating Terraform configuration files.
- `setup_google_pay_spring_lambda.py`: Script for setting up backend Spring Boot application with Kafka integration.
- `initial_scripts_for_project.py`: Script to automate the creation of necessary YAML files and provides instructions for manual tasks.

#### Instructions

##### Istio Installation

To install Istio, use the Istio CLI or Helm chart. For example, using Istioctl:

```sh
$ istioctl install --set profile=default -y
```

Enable Istio injection for the default namespace:

```sh
$ kubectl label namespace default istio-injection=enabled
```

##### TLS/SSL Configuration

To configure TLS/SSL for all services, ensure your Ingress or Gateway resources are configured with TLS certificates. You may use cert-manager for automatic certificate provisioning:

```sh
kubectl create namespace cert-manager
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager --namespace cert-manager --create-namespace --set installCRDs=true
```

#### Usage

Run the `initial_scripts_for_project.py` script to generate the YAML files for Jenkins deployment and API Gateway Horizontal Pod Autoscaler:

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

Run the `setup_k8s.py` script for additional setup:

```sh
python setup_k8s.py
```

Run with advanced options:

```sh
python k8s_management.py --install-istio --setup-jenkins --setup-monitoring
```

#### Summary

The scripts in the `k8s` folder provide a starting point for setting up CI/CD pipelines and managing scalability in your Kubernetes cluster. Follow the instructions for manual tasks and use the generated YAML files to automate deployments.

#### Additional Resources

For more information on the tools and concepts mentioned in this README, refer to the following resources:

- [Istio Documentation](https://istio.io/latest/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [Helm Documentation](https://helm.sh/docs/)
- [cert-manager Documentation](https://cert-manager.io/docs/)
- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [Kubernetes Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [YAML Language Reference](https://yaml.org/spec/1.2/spec.html)
