import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Function to draw boxes and arrows
def draw_box(ax, text, xy, width=2.5, height=1.0, box_color="lightblue"):
    box = FancyBboxPatch(xy, width, height,
                         boxstyle="round,pad=0.2",
                         edgecolor="black", facecolor=box_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=9, color="black", weight="bold")


def draw_arrow(ax, start_point, end_point, arrow_color="black"):
    ax.annotate("",
                xy=end_point, xycoords="data",
                xytext=start_point, textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color=arrow_color, linewidth=1.5))

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

# Draw components of the Data Fabric architecture:

# Data Sources Layer
draw_box(ax, "Data Sources\n(Databases, Files,\nAPIs, IoT)", xy=(0.5, 5.0), box_color="lightgreen")
draw_box(ax, "Legacy Systems", xy=(0.5, 3.8), box_color="lightgreen")
draw_box(ax, "Cloud Data", xy=(0.5, 2.6), box_color="lightgreen")
draw_arrow(ax, (3.0, 5.5), (4.0, 5.5), arrow_color="blue")
draw_arrow(ax, (3.0, 4.3), (4.0, 4.3), arrow_color="blue")
draw_arrow(ax, (3.0, 3.1), (4.0, 3.1), arrow_color="blue")

# Metadata Management Layer
draw_box(ax, "Metadata Management\n(Data Catalog, Lineage,\nSemantic Search)", xy=(4.5, 4.3), box_color="lightyellow")
draw_arrow(ax, (6.0, 4.8), (7.0, 4.8), arrow_color="blue")
draw_arrow(ax, (6.0, 4.2), (7.0, 4.2), arrow_color="blue")

# AI/ML Processor Layer
draw_box(ax, "AI/ML Integration\n(Data Analysis,\nPattern Recognition)", xy=(7.5, 4.8), box_color="lightcoral")

# Governance Layer
draw_box(ax, "Data Governance\n(Security, Privacy,\nCompliance)", xy=(4.5, 2.6), box_color="lightpink")
draw_arrow(ax, (6.0, 3.1), (7.0, 3.1), arrow_color="blue")

# Real-Time Processing Layer
draw_box(ax, "Real-Time Processing\n(Event Streaming,\nData Pipelines)", xy=(7.5, 3.4), box_color="lightblue")
draw_arrow(ax, (9.0, 4.4), (9.0, 4.0), arrow_color="blue")

# Data Consumers Layer
draw_box(ax, "Data Consumers\n(Dashboard Analytics,\nApplications, ML Models)", xy=(8.5, 1.0), box_color="lightgray")
draw_arrow(ax, (9.0, 3.8), (8.8, 2.5), arrow_color="blue")

# Add labels for the pipelines
ax.text(5.5, 6.2, "Data Fabric Processing Pipeline", fontsize=11, weight="bold", ha="center", color="darkblue")

# Adjust spacing
plt.tight_layout()
plt.show()
