/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package contest_2ima20.core.schematrees;

import nl.tue.geometrycore.geometry.linear.LineSegment;
import nl.tue.geometrycore.graphs.simple.SimpleVertex;

/**
 *
 * @author wmeulema
 */
public class Position extends SimpleVertex<LineSegment, Position, Edge> {

    public Node node;

    public int x() {
        return (int) Math.round(getX());

    }

    public int y() {
        return (int) Math.round(getY());
    }
}
