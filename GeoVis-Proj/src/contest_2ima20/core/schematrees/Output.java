/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contest_2ima20.core.schematrees;

import contest_2ima20.core.problem.Solution;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import nl.tue.geometrycore.algorithms.dsp.BreadthFirstSearch;
import nl.tue.geometrycore.datastructures.quadtree.PointQuadTree;
import nl.tue.geometrycore.geometry.linear.LineSegment;
import nl.tue.geometrycore.geometry.linear.Rectangle;
import nl.tue.geometrycore.geometryrendering.GeometryRenderer;
import nl.tue.geometrycore.geometryrendering.glyphs.PointStyle;
import nl.tue.geometrycore.geometryrendering.styling.Dashing;
import nl.tue.geometrycore.geometryrendering.styling.ExtendedColors;
import nl.tue.geometrycore.geometryrendering.styling.SizeMode;
import nl.tue.geometrycore.util.IntegerUtil;

/**
 *
 * @author Wouter Meulemans (w.meulemans@tue.nl)
 */
public class Output extends Solution {

    public Input input;
    public List<Graph> graphs;

    public Output(Input input) {
        this.input = input;
        this.graphs = new ArrayList();
        while (graphs.size() < input.sets.size()) {
            graphs.add(new Graph(input.sets.get(graphs.size())));
        }
    }

    @Override
    public boolean isValid() {

        // test unique locations in grid
        PointQuadTree<Position> pqt = new PointQuadTree(new Rectangle(-1, input.width + 1, -1, input.height + 1), 7);
        for (Graph g : graphs) {
            for (Position p : g.getVertices()) {
                // out of bounds?
                if (!IntegerUtil.inClosedInterval(p.x(), 0, input.width)) {
                    return false;
                }
                if (!IntegerUtil.inClosedInterval(p.y(), 0, input.height)) {
                    return false;
                }
                // too distant?
                if (Math.abs(p.x() - p.node.x()) + Math.abs(p.y() - p.node.y()) > input.radius) {
                    return false;
                }
                // overlap earlier position?
                if (pqt.find(p, 0.1) != null) {
                    return false;
                }
                // position OK
                pqt.insert(p);
            }
        }

        for (Graph g : graphs) {
            if (g.getEdges().size() < g.set.size() - 1) {
                // too small for a spanning graph
                return false;
            }

            // test spanning graph
            BreadthFirstSearch<Graph, LineSegment, Position, Edge> bfs = new BreadthFirstSearch(g);
            Position src = g.getVertices().get(0);
            bfs.run(src);
            for (Position p : g.getVertices()) {
                if (p != src && bfs.getPrevious(p) == null) {
                    return false;
                }
            }
        }

        return true;
    }

    @Override
    public double computeQuality() {
        int count = 0;
        List<Event> events = new ArrayList();
        for (Graph g : graphs) {
            for (Edge e : g.getEdges()) {
                Position left, right;
                if (e.getStart().x() <= e.getEnd().x()) {
                    left = e.getStart();
                    right = e.getEnd();
                } else {
                    left = e.getEnd();
                    right = e.getStart();
                }

                events.add(new Event(e, true, 2 * left.x()));
                events.add(new Event(e, false, 2 * right.x() + 1)); // puts endevents after start events
            }
        }

        events.sort((a, b) -> Integer.compare(a.x, b.x));

        List<Edge> state = new ArrayList();
        for (Event evt : events) {
            if (evt.start) {

                for (Edge e : state) {
                    if (e.getCommonVertex(evt.edge) == null && !e.getGeometry().intersect(evt.edge.getGeometry()).isEmpty()) {
                        count++;
                    }
                }

                state.add(evt.edge);
            } else {
                state.remove(evt.edge);
            }
        }

        return count;
    }

    private class Event {

        Edge edge;
        boolean start;
        int x;

        public Event(Edge e, boolean start, int x) {
            this.edge = e;
            this.start = start;
            this.x = x;
        }

    }

    @Override
    public void write(Writer out) throws IOException {
        for (Graph g : graphs) {
            for (Position p : g.getVertices()) {
                if (p.getGraphIndex() > 0) {
                    out.write(" ");
                }
                out.write(p.x() + ";" + p.y());
            }
            out.write("\n");
            for (Edge e : g.getEdges()) {
                if (e.getGraphIndex() > 0) {
                    out.write(" ");
                }
                out.write(e.getStart().getGraphIndex() + ";" + e.getEnd().getGraphIndex());
            }
            out.write("\n");
        }
    }

    @Override
    public void draw(GeometryRenderer render) {
        render.setSizeMode(SizeMode.VIEW);
        
        render.setStroke(ExtendedColors.black, 2, Dashing.dotted(2));
        for (Graph g : graphs) {
            for (Position p : g.getVertices()) {
                if (!p.isApproximately(p.node)) {
                    render.draw(new LineSegment(p, p.node));
                }
            }
        }

        input.draw(render);

        render.setPointStyle(PointStyle.CIRCLE_SOLID, 4);
        for (Graph g : graphs) {
            render.setStroke(input.getColor(g.set), 4, Dashing.SOLID);
            render.draw(g.getEdges());
            render.draw(g.getVertices());
        }
    }

    @Override
    public Rectangle getBoundingBox() {
        return input.getBoundingBox();
    }

    public void sortToInput() {
        graphs.sort((a, b) -> Integer.compare(a.set.id, b.set.id));
        for (Graph g : graphs) {
            g.sortVertices((a, b) -> Integer.compare(a.node.id, b.node.id));
        }
    }
}
