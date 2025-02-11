import yaml
import os
import subprocess

def write_yaml(file_name, content):
    with open(file_name, 'w') as file:
        yaml.safe_dump(content, file, default_flow_style=False)

def install_istio():
    print("\nStarting Istio installation...")
    profile = input("Enter the Istio profile to use (default/recommended): ")
    subprocess.run(["istioctl", "install", "--set", f"profile={profile}", "-y"])
    namespace = input("Enter the Kubernetes namespace to enable Istio injection (default): ")
    subprocess.run(["kubectl", "label", "namespace", namespace, "istio-injection=enabled"])

def setup_jenkins():
    print("\nSetting up Jenkins...")
    jenkins_yaml = {
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
    write_yaml('jenkins-deployment.yaml', jenkins_yaml)
    subprocess.run(["kubectl", "apply", "-f", "jenkins-deployment.yaml"])

def setup_hpa():
    print("\nConfiguring Horizontal Pod Autoscaler (HPA)...")
    hpa_yaml = {
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
    write_yaml('api-gateway-hpa.yaml', hpa_yaml)
    subprocess.run(["kubectl", "apply", "-f", "api-gateway-hpa.yaml"])

def setup_tls():
    print("\nSetting up SSL/TLS configuration...")
    subprocess.run(["kubectl", "create", "namespace", "cert-manager"])
    subprocess.run(["helm", "repo", "add", "jetstack", "https://charts.jetstack.io"])
    subprocess.run(["helm", "repo", "update"])
    subprocess.run(["helm", "install", "cert-manager", "jetstack/cert-manager", "--namespace", "cert-manager", "--create-namespace", "--set", "installCRDs=true"])

def main():
    print("Welcome to the Kubernetes Setup Assistant!")
    if input("Do you want to install Istio? (yes/no): ") == "yes":
        install_istio()
    if input("Do you want to set up Jenkins for CI/CD? (yes/no): ") == "yes":
        setup_jenkins()
    if input("Do you want to configure Horizontal Pod Autoscaler for an API gateway? (yes/no): ") == "yes":
        setup_hpa()
    if input("Do you want to configure SSL/TLS with cert-manager? (yes/no): ") == "yes":
        setup_tls()
    print("Setup complete. Thank you for using the Kubernetes Setup Assistant!")

if __name__ == "__main__":
    main()
