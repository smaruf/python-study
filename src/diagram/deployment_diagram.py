from diagrams import Cluster, Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS, Dynamodb
from diagrams.aws.network import ELB
from diagrams.aws.integration import SQS
from diagrams.onprem.database import Redis
from diagrams.onprem.client import Users

with Diagram("Gaming System - Deployment Diagram", show=False):
    player_client = Users("Game Client (Mobile/Desktop)")

    with Cluster("AWS Cloud"):
        lb = ELB("Load Balancer")
        with Cluster("Backend Services"):
            player_management = EC2("Player Management")
            matchmaking = EC2("Matchmaking")
            game_state = EC2("Game State")
            leaderboard = EC2("Leaderboard")
            analytics = EC2("Analytics")

        # Databases in the cluster
        with Cluster("Data Layer"):
            player_db = RDS("Player Database")
            state_cache = Redis("Game State Cache")
            leaderboard_db = Dynamodb("Leaderboard Database")

        # Linking the flow
        player_client >> lb >> [player_management, matchmaking, game_state, leaderboard, analytics]
        player_management >> player_db
        game_state >> state_cache
        leaderboard >> leaderboard_db
