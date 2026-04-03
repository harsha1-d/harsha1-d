/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package contest_2ima20.core.schematrees;

import nl.tue.geometrycore.geometry.Vector;

/**
 *
 * @author wmeulema
 */
public class Node extends Vector {

    public final int id;

    public Node(int id, double x, double y) {
        super(x, y);
        this.id = id;
    }

    public int x() {
        return (int) Math.round(getX());

    }

    public int y() {
        return (int) Math.round(getY());
    }

}
