from diagrams import Diagram, Cluster
from diagrams.aws.storage import S3
from diagrams.aws.database import RDS, Dynamodb
from diagrams.aws.analytics import Glue, Athena, KinesisDataAnalytics
from diagrams.aws.ml import Sagemaker
from diagrams.aws.management import Cloudwatch
from diagrams.aws.integration import Eventbridge
from diagrams.aws.security import IAM
from diagrams.general.workflow import Workflow

with Diagram("AWS Data Fabric Architecture", show=True, direction="LR"):

    # Data Sources
    with Cluster("Data Sources"):
        s3_data = S3("Object Storage (S3)")
        relational_db = RDS("Relational Database (RDS)")
        dynamo_db = Dynamodb("NoSQL Database (DynamoDB)")

    # Data Integration & Metadata Management
    with Cluster("Metadata & Integration"):
        glue = Glue("AWS Glue ETL & Catalog")
        athena = Athena("Athena Query Engine")
        metadata_mgmt = Workflow("Metadata Management")

    # Real-Time Processing
    with Cluster("Real-Time Processing"):
        kinesis_analytics = KinesisDataAnalytics("Kinesis Data Analytics")
        eventbridge = Eventbridge("Event Bridge")

    # AI/ML Integration
    with Cluster("AI/ML"):
        sagemaker = Sagemaker("Amazon SageMaker")

    # Governance & Security
    with Cluster("Governance & Security"):
        cloudwatch = Cloudwatch("Monitoring (CloudWatch)")
        iam = IAM("Access Management (IAM)")

    # Data Consumers
    with Cluster("Data Consumers"):
        dashboards = Workflow("Analytics Dashboards")
        business_apps = Workflow("Business Applications")
        ml_models = Workflow("ML Model Serving")

    # Connections
    relational_db >> glue
    dynamo_db >> glue
    s3_data >> glue
    glue >> metadata_mgmt >> athena
    metadata_mgmt >> kinesis_analytics
    kinesis_analytics >> eventbridge
    eventbridge >> dashboards
    eventbridge >> business_apps
    sagemaker >> ml_models
    cloudwatch >> dashboards
    iam >> [metadata_mgmt, eventbridge]
