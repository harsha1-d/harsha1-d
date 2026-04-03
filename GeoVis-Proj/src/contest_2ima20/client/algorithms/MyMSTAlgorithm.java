package contest_2ima20.client.algorithms;

import contest_2ima20.client.schematrees.SchematicTreesAlgorithm;
import contest_2ima20.core.schematrees.*;
import java.util.*;

/**
connects each NodeSet using a Minimum Spanning Tree (MST) to minimize total edge length.
 */
public class MyMSTAlgorithm extends SchematicTreesAlgorithm {

    @Override
    public Output doAlgorithm(Input input) {
        Output output = new Output(input);

        // Process each set (Hyperedge) individually
        for (Graph graph : output.graphs) {
            NodeSet set = graph.set;
            if (set.size() < 2) continue;

            // Create a mapping from Node to Position for quick lookup
            Map<Node, Position> nodeToPosition = new HashMap<>();
            for (Position p : graph.getVertices()) {
                nodeToPosition.put(p.node, p);
            }

            // Generate all possible edges within this set for Kruskal's
            List<PotentialEdge> potentialEdges = new ArrayList<>();
            for (int i = 0; i < set.size(); i++) {
                for (int j = i + 1; j < set.size(); j++) {
                    Node u = set.get(i);
                    Node v = set.get(j);
                    double dist = Math.hypot(u.getX() - v.getX(), u.getY() - v.getY());
                    potentialEdges.add(new PotentialEdge(u, v, dist));
                }
            }

            // Sort potential edges by distance
            Collections.sort(potentialEdges);

            // Union-Find (DSU) to prevent cycles within the set
            Map<Node, Node> parent = new HashMap<>();
            for (Node n : set) parent.put(n, n);

            int edgesCount = 0;
            for (PotentialEdge pe : potentialEdges) {
                if (edgesCount >= set.size() - 1) break;

                Node rootU = find(parent, pe.u);
                Node rootV = find(parent, pe.v);

                if (rootU != rootV) {
                    // Add the edge to the graph for this specific set
                    Position posU = nodeToPosition.get(pe.u);
                    Position posV = nodeToPosition.get(pe.v);
                    graph.addEdge(posU, posV);
                    parent.put(rootU, rootV);
                    edgesCount++;
                }
            }
        }

        return output;
    }

    // Standard Find operation for Disjoint Set Union
    private Node find(Map<Node, Node> parent, Node i) {
        if (parent.get(i).equals(i)) return i;
        Node root = find(parent, parent.get(i));
        parent.put(i, root); // Path compression
        return root;
    }

    // Helper class for Kruskal's sorting
    private static class PotentialEdge implements Comparable<PotentialEdge> {
        Node u, v;
        double distance;

        PotentialEdge(Node u, Node v, double distance) {
            this.u = u;
            this.v = v;
            this.distance = distance;
        }

        @Override
        public int compareTo(PotentialEdge other) {
            return Double.compare(this.distance, other.distance);
        }
    }

}