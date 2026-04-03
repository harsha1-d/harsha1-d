/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package contest_2ima20.core.schematrees;

import nl.tue.geometrycore.geometry.linear.LineSegment;
import nl.tue.geometrycore.graphs.simple.SimpleGraph;

/**
 *
 * @author wmeulema
 */
public class Graph extends SimpleGraph<LineSegment,Position,Edge> {
    
    public final NodeSet set;

    public Graph(NodeSet set) {
        this.set = set;
        
        for (Node n : set) {
            addVertex(n).node = n;
        }
    }
    
    public void addEdge(Position from, Position to) {
        addEdge(from, to, new LineSegment(from.clone(), to.clone()));
    }
}
