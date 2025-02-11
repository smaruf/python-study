import subprocess
import logging
import yaml
import sys
import shlex
from datetime import datetime
from argparse import ArgumentParser

def setup_logger():
    """
    Setup logging configuration to output both to stdout and a log file.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[
                            logging.FileHandler("k8s_management.log"),
                            logging.StreamHandler(sys.stdout)
                        ])

def write_yaml(file_name, data):
    """
    Utility function to write data to a YAML file.
    
    Args:
    file_name (str): The name of the file to write.
    data (dict): The data to write to the file.
    """
    with open(file_name, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def install_istio(profile='default'):
    """
    Install Istio using the istioctl command with a specified profile.
    
    Args:
    profile (str): The installation profile name.
    """
    logging.info("Installing Istio with profile: %s", profile)
    subprocess.run(["istioctl", "install", "--set", f"profile={profile}", "--skip-confirmation"], check=True)
    logging.info("Istio installation completed.")

def setup_jenkins():
    """
    Setup Jenkins deployment on the Kubernetes cluster.
    """
    logging.info("Setting up Jenkins...")
    jenkins_yaml = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {'name': 'jenkins'},
        'spec': {
            'replicas': 1,
            'selector': {'matchLabels': {'app': 'jenkins'}},
            'template': {
                'metadata': {'labels': {'app': 'jenkins'}},
                'spec': {
                    'containers': [{
                        'name': 'jenkins',
                        'image': 'jenkins/jenkins:lts',
                        'ports': [{'containerPort': 8080}]
                    }]
                }
            }
        }
    }
    write_yaml('jenkins_deployment.yaml', jenkins_yaml)
    subprocess.run(["kubectl", "apply", "-f", "jenkins_deployment.yaml"], check=True)
    logging.info("Jenkins deployment completed.")

def setup_monitoring():
    """
    Setup monitoring using Prometheus and Grafana on the Kubernetes cluster.
    """
    logging.info("Setting up monitoring tools...")
    subprocess.run(["helm", "repo", "add", "prometheus-community", "https://prometheus-community.github.io/helm-charts"], check=True)
    subprocess.run(["helm", "install", "kube-prometheus-stack", "prometheus-community/kube-prometheus-stack"], check=True)
    logging.info("Monitoring tools setup completed.")

def fetch_cpu_usage():
    """
    Fetch average CPU utilization across nodes.
    
    Returns:
    float: The average CPU utilization in percentage.
    """
    command = "kubectl top nodes --no-headers | awk '{print $3}' | sed 's/%//g' | awk '{s+=$1} END {print s/NR}'"
    result = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def setup_predictive_scaling():
    """
    Set up predictive scaling based on CPU usage.
    """
    logging.info("Setting up predictive scaling...")
    try:
        cpu_usage = fetch_cpu_usage()
        if cpu_usage > 80:
            logging.info("High CPU usage detected (%s%%). Scaling up...", cpu_usage)
            subprocess.run(["kubectl", "scale", "deployment/my-app", "--replicas=10"], check=True)
        elif cpu_usage < 40:
            logging.info("Low CPU usage detected (%s%%). Scaling down...", cpu_usage)
            subprocess.run(["kubectl", "scale", "deployment/my-app", "--replicas=2"], check=True)
        else:
            logging.info("CPU usage is normal (%s%%). No scaling performed.", cpu_usage)
    except Exception as e:
        logging.error("Failed to perform predictive scaling: %s", str(e))
    logging.info("Predictive scaling setup completed.")

def adjust_nginx_configuration():
    """
    Adjust Nginx configuration based on the time of day, setting more workers during peak hours.
    """
    now = datetime.now()
    if 9 <= now.hour < 17:  # During business hours
        workers = 10
    else:  # Off business hours
        workers = 3
        
    nginx_config = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
data:
  nginx.conf: |
    user nginx;
    worker_processes {workers};
    error_log /var/log/nginx/error.log warn;
    pid /var/run/nginx.pid;
    events {{ worker_connections 1024; }}
    http {{ include /etc/nginx/mime.types; default_type application/octet-stream; }}
    """
    write_yaml("nginx-config.yaml", yaml.safe_load(nginx_config))
    subprocess.run(["kubectl", "apply", "-f", "nginx-config.yaml"], check=True)
    logging.info("Adjusted Nginx configurations for current time: %s workers", workers)

def dynamic_configuration():
    """
    Apply dynamic configuration adjustments based on changing factors such as time of day.
    """
    logging.info("Applying dynamic configuration adjustments...")
    try:
        adjust_nginx_configuration()
    except Exception as e:
        logging.error("Failed to apply dynamic configuration adjustments: %s", str(e))
    logging.info("Dynamic configuration adjustments applied.")

def parse_arguments():
    """
    Parse command line arguments to determine actions.
    
    Returns:
    Namespace: The arguments namespace.
    """
    parser = ArgumentParser(description="Advanced Kubernetes Management Script")
    parser.add_argument('--install-istio', action='store_true', help="Install Istio service mesh")
    parser.add_argument('--setup-jenkins', action='store_true', help="Deploy Jenkins for CI/CD")
    parser.add_argument('--setup-monitoring', action='store_true', help="Setup Prometheus and Grafana for monitoring")
    parser.add_argument('--predictive-scaling', action='store_true', help="Setup predictive scaling")
    parser.add_argument('--dynamic-config', action='store_true', help="Apply dynamic configuration adjustments")
    return parser.parse_args()

def main():
    """
    Main function to orchestrate Kubernetes management operations based on provided arguments.
    """
    setup_logger()
    args = parse_arguments()
    
    if args.install_istio:
        install_istio()
    if args.setup_jenkins:
        setup_jenkins()
    if args.setup_monitoring:
        setup_monitoring()
    if args.predictive_scaling:
        setup_predictive_scaling()
    if args.dynamic_config:
        dynamic_configuration()

    logging.info("Kubernetes management operations completed.")

if __name__ == "__main__":
    main()
