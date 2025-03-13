from diagrams import Diagram, Cluster
from diagrams.azure.storage import BlobStorage, DataLake
from diagrams.azure.database import SQLDatabase, CosmosDb
from diagrams.azure.analytics import SynapseAnalytics, DataFactory, Databricks
from diagrams.azure.integration import EventGrid, ServiceBus
from diagrams.azure.ml import MachineLearning
from diagrams.azure.monitor import Monitor
from diagrams.azure.security import Sentinel, ActiveDirectory
from diagrams.azure.web import AppServices
from diagrams.onprem.workflow import Workflow

with Diagram("Azure Data Fabric Architecture", show=True, direction="LR"):

    # Data Sources
    with Cluster("Data Sources"):
        blob_storage = BlobStorage("Blob Storage")
        data_lake = DataLake("Data Lake Storage")
        sql_database = SQLDatabase("SQL Database")
        cosmos_db = CosmosDb("CosmosDB (NoSQL)")

    # Data Integration and Metadata Management
    with Cluster("Data Integration & Metadata"):
        data_factory = DataFactory("Azure Data Factory")
        synapse = SynapseAnalytics("Synapse Analytics")
        databricks = Databricks("Azure Databricks")
        metadata_mgmt = Workflow("Metadata Management")

    # Real-Time Processing
    with Cluster("Real-Time Processing"):
        event_grid = EventGrid("Event Grid")
        service_bus = ServiceBus("Service Bus")

    # AI/ML Integration
    with Cluster("AI/ML"):
        ml_service = MachineLearning("Azure Machine Learning")

    # Governance & Security
    with Cluster("Governance & Security"):
        monitor = Monitor("Azure Monitor")
        ad_security = ActiveDirectory("Azure Active Directory")
        sentinel = Sentinel("Azure Sentinel")

    # Data Consumers
    with Cluster("Data Consumers"):
        analytics_dashboard = Workflow("Power BI Dashboards")
        business_app = AppServices("Custom Business Applications")
        ml_model_serving = Workflow("ML Model Serving")

    # Connect components
    # Data Sources to Data Integration
    blob_storage >> data_factory
    data_lake >> data_factory
    sql_database >> synapse
    cosmos_db >> synapse

    # Data Factory Pipeline
    data_factory >> metadata_mgmt >> databricks

    # Real-Time Processing
    metadata_mgmt >> service_bus
    service_bus >> event_grid

    # AI/ML and Data Consumption
    event_grid >> analytics_dashboard
    event_grid >> business_app
    databricks >> ml_service >> ml_model_serving

    # Governance and Security
    ad_security >> [data_factory, synapse, service_bus, ml_service]
    monitor >> [analytics_dashboard, business_app, ml_model_serving]
    sentinel >> [metadata_mgmt, ad_security]
