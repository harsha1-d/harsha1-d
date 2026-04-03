/*
 * GeometryCore library   
 * Copyright (C) 2019   Wouter Meulemans (w.meulemans@tue.nl)
 * 
 * Licensed under GNU GPL v3. See provided license documents (license.txt and gpl-3.0.txt) for more information.
 */
package nl.tue.geometrycore.algorithms.delaunay;

import java.util.Iterator;
import nl.tue.geometrycore.geometry.Vector;
import nl.tue.geometrycore.geometry.linear.LineSegment;
import nl.tue.geometrycore.graphs.simple.SimpleEdge;
import nl.tue.geometrycore.graphs.simple.SimpleGraph;
import nl.tue.geometrycore.graphs.simple.SimpleVertex;
import nl.tue.geometrycore.util.Pair;

/**
 * Convenience class for easily computing the Delaunay triangulation of a set of
 * points.
 *
 * @author Wouter Meulemans (w.meulemans@tue.nl)
 * @param <TVector> Actual class of the points being provided
 */
public class SimpleDelaunayTriangulation<TVector extends Vector> {

    private final IGraph graph = new IGraph();

    /**
     * Computes the Delaunay triangulation of the given set of points.
     *
     * @param points
     */
    public void run(Iterable<TVector> points) {
        this.graph.clear();
        for (TVector p : points) {
            IVertex v = graph.addVertex(p);
            v.original = p;
        }
        DelaunayTriangulation<IGraph, LineSegment, IVertex, IEdge> dt = new DelaunayTriangulation<>(graph, (LineSegment geometry) -> geometry);
        dt.run();
    }

    /**
     * The edges of the Delaunay triangulation, given as pairs of objects, being
     * the original points provided. Only edges from the last call to run() are
     * provided.
     *
     * @return Iterable for the edges in the Delaunay triangulation
     */
    public Iterable<Pair<TVector, TVector>> edges() {
        Iterator<IEdge> it = graph.getEdges().iterator();
        return () -> {
            return new Iterator() {
                @Override
                public boolean hasNext() {
                    return it.hasNext();
                }

                @Override
                public Object next() {
                    IEdge e = it.next();
                    return new Pair(e.getStart().original, e.getEnd().original);
                }
            };
        };
    }

    private class IGraph extends SimpleGraph<LineSegment, IVertex, IEdge> {

        public IGraph() {
            super(false);
        }

        @Override
        public IEdge createEdge() {
            return new IEdge();
        }

        @Override
        public IVertex createVertex() {
            return new IVertex();
        }

    }

    private class IVertex extends SimpleVertex<LineSegment, IVertex, IEdge> {

        private TVector original;
    }

    private class IEdge extends SimpleEdge<LineSegment, IVertex, IEdge> {

    }
}
