#!/usr/bin/env python3
"""
Level 4 - Expert: Kubernetes Deployment Automation
"""

import subprocess
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional

class KubernetesManager:
    """Manages Kubernetes deployments"""
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.check_kubectl()
    
    def check_kubectl(self):
        """Check if kubectl is available"""
        try:
            subprocess.run(
                ["kubectl", "version", "--client"],
                capture_output=True,
                check=True
            )
            print("✓ kubectl is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ kubectl not found - running in demo mode")
    
    def create_deployment(self, name: str, image: str, replicas: int = 3, port: int = 8080):
        """Create a Kubernetes deployment"""
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': name,
                'namespace': self.namespace,
                'labels': {'app': name}
            },
            'spec': {
                'replicas': replicas,
                'selector': {'matchLabels': {'app': name}},
                'template': {
                    'metadata': {'labels': {'app': name}},
                    'spec': {
                        'containers': [{
                            'name': name,
                            'image': image,
                            'ports': [{'containerPort': port}],
                            'resources': {
                                'requests': {'memory': '128Mi', 'cpu': '100m'},
                                'limits': {'memory': '256Mi', 'cpu': '200m'}
                            },
                            'livenessProbe': {
                                'httpGet': {'path': '/health', 'port': port},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/ready', 'port': port},
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Save to file
        deployment_file = f"{name}-deployment.yaml"
        with open(deployment_file, 'w') as f:
            yaml.dump(deployment, f, default_flow_style=False)
        
        print(f"✓ Created deployment manifest: {deployment_file}")
        return deployment_file
    
    def create_service(self, name: str, port: int = 8080, target_port: int = 8080):
        """Create a Kubernetes service"""
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': name,
                'namespace': self.namespace
            },
            'spec': {
                'type': 'LoadBalancer',
                'selector': {'app': name},
                'ports': [{
                    'protocol': 'TCP',
                    'port': port,
                    'targetPort': target_port
                }]
            }
        }
        
        # Save to file
        service_file = f"{name}-service.yaml"
        with open(service_file, 'w') as f:
            yaml.dump(service, f, default_flow_style=False)
        
        print(f"✓ Created service manifest: {service_file}")
        return service_file
    
    def apply_manifest(self, manifest_file: str):
        """Apply a Kubernetes manifest"""
        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", manifest_file, "-n", self.namespace],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Applied manifest: {manifest_file}")
            print(result.stdout)
        except FileNotFoundError:
            print(f"Demo: kubectl apply -f {manifest_file} -n {self.namespace}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to apply manifest: {e.stderr}")
    
    def scale_deployment(self, name: str, replicas: int):
        """Scale a deployment"""
        try:
            result = subprocess.run(
                ["kubectl", "scale", "deployment", name, 
                 f"--replicas={replicas}", "-n", self.namespace],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Scaled {name} to {replicas} replicas")
            print(result.stdout)
        except FileNotFoundError:
            print(f"Demo: kubectl scale deployment {name} --replicas={replicas} -n {self.namespace}")
    
    def rollout_status(self, name: str):
        """Check rollout status"""
        try:
            result = subprocess.run(
                ["kubectl", "rollout", "status", f"deployment/{name}", 
                 "-n", self.namespace],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Rollout status for {name}")
            print(result.stdout)
        except FileNotFoundError:
            print(f"Demo: kubectl rollout status deployment/{name} -n {self.namespace}")
    
    def get_pods(self, label: str):
        """Get pods by label"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-l", label, 
                 "-n", self.namespace, "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            pods = json.loads(result.stdout)
            print(f"✓ Found {len(pods.get('items', []))} pods")
            return pods
        except FileNotFoundError:
            print(f"Demo: kubectl get pods -l {label} -n {self.namespace}")
            return None

def main():
    """Main execution"""
    
    print("="*60)
    print("Kubernetes Deployment Automation")
    print("="*60)
    
    # Initialize manager
    k8s = KubernetesManager(namespace="production")
    
    # Application details
    app_name = "my-app"
    image = "nginx:latest"
    replicas = 3
    port = 80
    
    print(f"\nDeploying application: {app_name}")
    print(f"Image: {image}")
    print(f"Replicas: {replicas}")
    print()
    
    # Create deployment manifest
    deployment_file = k8s.create_deployment(app_name, image, replicas, port)
    
    # Create service manifest
    service_file = k8s.create_service(app_name, port, port)
    
    print("\n" + "="*60)
    print("Applying manifests to cluster")
    print("="*60)
    
    # Apply manifests
    k8s.apply_manifest(deployment_file)
    k8s.apply_manifest(service_file)
    
    print("\n" + "="*60)
    print("Checking deployment status")
    print("="*60)
    
    # Check rollout status
    k8s.rollout_status(app_name)
    
    # Get pods
    k8s.get_pods(f"app={app_name}")
    
    print("\n✓ Kubernetes deployment automation completed!")

if __name__ == "__main__":
    main()
