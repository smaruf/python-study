## Generate output components:

```
+--------------------+       +---------------------------+
|   Data Sources     | ----> |Data Integration & Metadata|
| Blob Storage       |       |(Data Factory, Synapse,    |
| Data Lake, CosmosDB|       | Databricks, Metadata Mgt) |
+--------------------+       +---------------------------+
                                   |       |
                          +--------+       +--------+
                          |                        |
              +--------------------+   +------------------------+
              | Real-Time Processing|   | AI/ML Integration      |
              | Event Grid, Service |   | Azure Machine Learning |
              | Bus                 |   +------------------------+
              |                        +--------+
                 |                             |
         +---------------+          +----------------------+
         |Governance &   |          | Data Consumers       |
         |Security (AD,  | --------> | Power BI, Apps, ML  |
         | Sentinel, Mon)|            | Models             |
         +---------------+            +--------------------+
```
