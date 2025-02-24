import os
import subprocess

# Create directories and files with specified content
def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)

# Define project structure and contents
project_structure = {
    "src/main/java/com/example/demo/DemoApplication.java": """package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}""",
    "src/main/java/com/example/demo/config/SecurityConfig.java": """// Security configuration content here""",
    "src/main/java/com/example/demo/controller/HelloController.java": """// Controller content here""",
    "src/main/java/com/example/demo/service/GrpcService.java": """// gRPC service content here""",
    "src/main/java/com/example/demo/grpc/HelloService.java": """// gRPC HelloService content here""",
    "src/main/java/com/example/demo/grpc/HelloServiceGrpc.java": """// gRPC HelloServiceGrpc content here""",
    "src/main/resources/application.properties": """# Application properties content here""",
    "src/test/java/com/example/demo/DemoApplicationTests.java": """package com.example.demo;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class DemoApplicationTests {

    @Test
    void contextLoads() {
    }

}""",
    "src/test/java/com/example/demo/controller/HelloControllerTest.java": """// Controller test content here""",
    "src/test/java/com/example/demo/service/GrpcServiceTest.java": """// gRPC service test content here""",
    "k8s/deployment.yaml": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: demo-deployment
spec:
  selector:
    matchLabels:
      app: demo
  replicas: 2
  template:
    metadata:
      labels:
        app: demo
    spec:
      containers:
      - name: demo
        image: demo-image
        ports:
        - containerPort: 8080
        - containerPort: 50051""",
    "k8s/service.yaml": """apiVersion: v1
kind: Service
metadata:
  name: demo-service
spec:
  selector:
    app: demo
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
    - protocol: TCP
      port: 50051
      targetPort: 50051""",
    "k8s/ingress.yaml": """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: demo-ingress
spec:
  rules:
  - host: demo.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: demo-service
            port:
              number: 80""",
    "k8s/runner.sh": """#!/bin/bash

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

echo "Kubernetes resources deployed successfully!"""",
    "load-tests/load-test-script.jmx": """// Load test script content here""",
    "Jenkinsfile": """pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'mvn clean package'
            }
        }
        stage('Unit Test') {
            steps {
                sh 'mvn test'
            }
        }
        stage('Integration Test') {
            steps {
                sh 'mvn verify -Pintegration'
            }
        }
        stage('Deploy to Kubernetes') {
            steps {
                sh './k8s/runner.sh'
            }
        }
    }
    post {
        always {
            junit 'target/surefire-reports/*.xml'
        }
    }
}"""
}

# Create project structure and files
for path, content in project_structure.items():
    create_file(path, content)

# Function to display menu and perform actions
def display_menu():
    print("Select an action:")
    print("1. Build Project")
    print("2. Deploy Project")
    print("3. Run Tests")
    print("4. Load Test")
    print("5. Exit")

def main():
    while True:
        display_menu()
        choice = input("Enter your choice: ")
        if choice == '1':
            subprocess.run(['mvn', 'clean', 'package'])
        elif choice == '2':
            subprocess.run(['./k8s/runner.sh'])
        elif choice == '3':
            subprocess.run(['mvn', 'test'])
            subprocess.run(['mvn', 'verify', '-Pintegration'])
        elif choice == '4':
            subprocess.run(['jmeter', '-n', '-t', 'load-tests/load-test-script.jmx', '-l', 'load-tests/results.jtl'])
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
