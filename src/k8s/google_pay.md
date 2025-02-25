## Generated project from `setup_google_pay_spring_lambda`

```
project-root/
│
├── backend/                     # Spring Boot Java application
│   ├── src/
│   │   ├── main/java/com/example/backend/
│   │   │   └── Application.java  # Spring Boot startup class
│   │   └── main/resources/
│   │       └── application.properties   # Configuration for Spring application
│   ├── test/
│   │   └── java/com/example/backend/    # Java tests (JUnit)
│   └── pom.xml                  # Maven configuration file
│
├── lambda/                      # AWS Lambda functions (Serverless)
│   └── token_creator.py         # Python Lambda for creating tokens
│
├── k8s/                         # Kubernetes configuration files
│   ├── backend_deployment.yaml  # K8s deployment for backend
│   └── service.yaml             # K8s service to expose the backend
│
├── terraform/                   # Terraform files for infrastructure
│   └── main.tf                  # Terraform configuration for AWS resources
│
├── opensearch/                  # OpenSearch configurations
│   └── dashboard_config.yml     # Configurations for OpenSearch Dashboards
│
├── ci_cd/                       # Continuous Integration and Deployment
│   └── Jenkinsfile              # Jenkins pipeline definition
│
└── test/                        # BDD and integration tests
    ├── features/                # Cucumber feature files
    │   └── backend_features.feature
    └── step_definitions/        # Python step definitions for BDD (using Behave)
        └── backend_steps.py
```
