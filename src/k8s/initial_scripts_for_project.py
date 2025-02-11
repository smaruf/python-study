import yaml

def write_yaml(file_name, content):
    with open(file_name, 'w') as file:
        yaml.safe_dump(content, file, default_flow_style=False)

def main():
    # Install Istio Instructions (Displayed in stdout for manual execution)
    istio_installation = """
To install Istio, you should use the Istio CLI or Helm chart.
For example, using Istioctl:
$ istioctl install --set profile=default -y
Enable Istio injection for default namespace:
$ kubectl label namespace default istio-injection=enabled
"""

    # Jenkins Deployment for CI/CD
    jenkins_deployment = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': 'jenkins',
            'namespace': 'ci-cd'
        },
        'spec': {
            'selector': {
                'matchLabels': {'app': 'jenkins'}
            },
            'replicas': 1,
            'template': {
                'metadata': {
                    'labels': {'app': 'jenkins'}
                },
                'spec': {
                    'containers': [{
                        'name': 'jenkins',
                        'image': 'jenkins/jenkins:lts',
                        'ports': [{'containerPort': 8080}],
                        'volumeMounts': [{
                            'name': 'jenkins-data',
                            'mountPath': '/var/jenkins_home'
                        }]
                    }],
                    'volumes': [{
                        'name': 'jenkins-data',
                        'persistentVolumeClaim': {
                            'claimName': 'jenkins-pvc'
                        }
                    }]
                }
            }
        }
    }

    # Horizontal Pod Autoscaler for API Gateway
    hpa = {
        'apiVersion': 'autoscaling/v2',
        'kind': 'HorizontalPodAutoscaler',
        'metadata': {
            'name': 'api-gateway-hpa',
            'namespace': 'default'
        },
        'spec': {
            'scaleTargetRef': {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'name': 'api-gateway-deployment'
            },
            'minReplicas': 2,
            'maxReplicas': 10,
            'metrics': [{
                'type': 'Resource',
                'resource': {
                    'name': 'cpu',
                    'target': {
                        'type': 'Utilization',
                        'averageUtilization': 50
                    }
                }
            }]
        }
    }

    # Configure TLS/SSL for secure communications (Dummy example)
    tls_configuration = """
To configure TLS/SSL for all services, ensure your Ingress or Gateway resources
are configured with TLS certificates. You may use cert-manager for automatic certificate provisioning:
$ kubectl create namespace cert-manager
$ helm repo add jetstack https://charts.jetstack.io
$ helm repo update
$ helm install cert-manager jetstack/cert-manager --namespace cert-manager --create-namespace --set installCRDs=true
"""

    # Print installation instructions for some manual tasks
    print(istio_installation)
    print(tls_configuration)

    # Write YAML files for automation parts
    write_yaml('jenkins-deployment.yaml', jenkins_deployment)
    write_yaml('api-gateway-hpa.yaml', hpa)

    print("Kubernetes YAML files for the API gateway and CI/CD pipeline have been created.")

if __name__ == "__main__":
    main()
