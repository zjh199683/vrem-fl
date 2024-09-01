import networkx as nx
import osmnx as ox

""" 

    To convert the osm into a sumo readable network run in the terminal

    netconvert --osm-files padova.osm -o padova.net.xml --geometry.remove --ramps.guess --junctions.join 
    --tls.guess-signals --tls.discard-simple --tls.join --tls.default-type actuated
    
    python /usr/share/sumo/tools/randomTrips.py -n padova.net.xml -o padova.trips.xml --random 
    --fringe-factor 10 -p 3.6
    
    duarouter -n padova.net.xml -o padova.rou.xml --route-files padova.trips.xml --ignore-errors

"""

# coordinates
north, east, south, west = 45.4635, 11.7797, 45.3507, 11.9916
# 12.5578 km x-axis; 23.5466 km y-axis (check)
# Downloading the map as a graph object
G = ox.graph_from_bbox(north, south, east, west, network_type='drive', simplify=True, clean_periphery=True)
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
# removes isolated nodes
G = ox.utils_graph.remove_isolated_nodes(G)
kph = nx.get_edge_attributes(G, 'speed_kph')
length = nx.get_edge_attributes(G, 'length')
# transform kph in mps
mps = dict()
for key, value in kph.items():
    mps[key] = float(value) / 3.6
nx.set_edge_attributes(G, mps, "speed_mps")
ox.save_graph_xml(G, filepath='padova.osm')
ox.save_graphml(G, filepath='padova.graphml')