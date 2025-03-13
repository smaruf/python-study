## Generated component output:

```
+--------------------+       +---------------------------+
|   Data Sources     | ----> |Data Integration & Metadata|
| GCS, Bigtable,     |       |(Dataflow, Dataproc,       |
| Spanner, IoT Core  |       | BigQuery, Metadata Mgt)   |
+--------------------+       +---------------------------+
                                   |       |
                          +--------+       +--------+
                          |                        |
              +--------------------+   +------------------------+
              | Real-Time Processing|   | AI/ML Integration      |
              | (Pub/Sub)           |   | AI Platform            |
              +--------------------+   +------------------------+
                             |                      |
                 +----------------------+  +----------------------+
                 | Data Consumers       |  | Governance           |
                 | Looker Dashboards,   |  | IAM, Cloud Monitoring|
                 | Apps, Models         |  +----------------------+
                 +----------------------+
```
