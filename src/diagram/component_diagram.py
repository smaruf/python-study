from diagrams import Diagram
from diagrams.aws.network import APIGateway
from diagrams.onprem.database import PostgreSQL, Redis, MongoDB
from diagrams.onprem.client import Users
from diagrams.generic.compute import Server
from diagrams.generic.database import Databases
from diagrams.custom import Custom

with Diagram("Gaming System - Component Diagram", show=False):
    player = Users("Player (Client)")
    api_gateway = APIGateway("REST API Gateway")

    # Backend Microservices
    player_management = Server("Player Management")
    matchmaking = Server("Matchmaking")
    game_state = Server("Game State")
    leaderboard = Server("Leaderboard")
    analytics = Server("Analytics")

    # Databases
    player_db = MongoDB("Player Database")
    state_cache = Redis("Game State Cache")
    leaderboard_db = PostgreSQL("Leaderboard Database")
    analytics_store = Databases("Analytics Cluster")

    # Player Flow
    player >> api_gateway >> [player_management, matchmaking, game_state, leaderboard, analytics]

    # Databases Associated
    player_management >> player_db
    game_state >> state_cache
    leaderboard >> leaderboard_db
    analytics >> analytics_store
