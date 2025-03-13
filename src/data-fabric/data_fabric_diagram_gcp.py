from diagrams import Diagram, Cluster
from diagrams.gcp.storage import GCS, BigTable, Spanner
from diagrams.gcp.analytics import BigQuery, Dataflow, PubSub, Dataproc
from diagrams.gcp.ml import AIPlatform
from diagrams.gcp.devtools import Monitoring
from diagrams.gcp.security import IAM
from diagrams.gcp.iot import IoTCore
from diagrams.custom import Custom

# Create the Data Fabric Architecture diagram
with Diagram("GCP Data Fabric Architecture", show=True, direction="LR"):

    # Data Sources
    with Cluster("Data Sources"):
        gcs_storage = GCS("Cloud Storage")
        bigtable = BigTable("Bigtable (NoSQL)")
        spanner = Spanner("Cloud Spanner (SQL)")
        iot_core = IoTCore("IoT Core")

    # Data Integration & Metadata Management
    with Cluster("Data Integration & Metadata"):
        dataflow = Dataflow("Dataflow (ETL & Stream)")
        dataproc = Dataproc("Dataproc (Hadoop/Spark)")
        bigquery = BigQuery("BigQuery (Analytics Engine)")
        metadata = Custom("Metadata Management", "icons/metadata.png")  # Add metadata representation

    # Real-Time Processing
    with Cluster("Real-Time Processing"):
        pubsub = PubSub("Pub/Sub (Event Streaming)")

    # AI/ML Integration
    with Cluster("AI/ML"):
        ai_platform = AIPlatform("AI Platform")

    # Governance & Security
    with Cluster("Governance & Monitoring"):
        iam = IAM("Identity & Access Management")
        monitoring = Monitoring("Cloud Monitoring")

    # Data Consumers
    with Cluster("Data Consumers"):
        dashboards = Custom("Looker Dashboards", "icons/looker.png")  # Replace with a Looker Dashboards icon
        business_apps = Custom("Business Applications", "icons/apps.png")
        ml_models = Custom("ML Model Serving", "icons/mlmodel.png")

    # Connections
    # Data Sources to Data Integration
    gcs_storage >> dataflow
    bigtable >> dataflow
    spanner >> dataproc
    iot_core >> pubsub

    # Data Integration Pipeline
    dataflow >> metadata >> bigquery >> dataproc

    # Real-Time Processing
    bigquery >> pubsub

    # AI/ML Integration
    bigquery >> ai_platform >> ml_models

    # Data Consumers
    pubsub >> dashboards
    pubsub >> business_apps

    # Governance & Monitoring
    iam >> [dataflow, pubsub, ai_platform]
    monitoring >> [dashboards, business_apps, ml_models]
