/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contest_2ima20.core.schematrees;

import contest_2ima20.core.problem.Problem;
import java.awt.Color;
import java.awt.geom.AffineTransform;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import nl.tue.geometrycore.geometry.Vector;
import nl.tue.geometrycore.geometry.linear.LineSegment;
import nl.tue.geometrycore.geometry.linear.Polygon;
import nl.tue.geometrycore.geometry.linear.Rectangle;
import nl.tue.geometrycore.geometryrendering.GeometryPanel;
import nl.tue.geometrycore.geometryrendering.GeometryRenderer;
import nl.tue.geometrycore.geometryrendering.glyphs.PointStyle;
import nl.tue.geometrycore.geometryrendering.styling.Dashing;
import nl.tue.geometrycore.geometryrendering.styling.ExtendedColors;
import nl.tue.geometrycore.geometryrendering.styling.Hashures;
import nl.tue.geometrycore.geometryrendering.styling.SizeMode;
import nl.tue.geometrycore.geometryrendering.styling.TextAnchor;
import nl.tue.geometrycore.util.DoubleUtil;

/**
 *
 * @author Wouter Meulemans (w.meulemans@tue.nl)
 */
public class Input extends Problem<Output> {

    public int width, height, radius;
    public List<NodeSet> sets;
    private Color[] colors = null;

    @Override
    public void read(BufferedReader read) throws IOException {
        sets = new ArrayList();

        String[] dim = read.readLine().split(" ");
        width = Integer.parseInt(dim[0]);
        height = Integer.parseInt(dim[1]);
        radius = Integer.parseInt(dim[2]);

        read.lines().forEach((L) -> {
            NodeSet set = new NodeSet(sets.size());
            sets.add(set);

            String[] setspec = L.split(" ");
            for (String pt : setspec) {
                String[] xy = pt.split(";");
                int x = Integer.parseInt(xy[0]);
                int y = Integer.parseInt(xy[1]);
                Node n = new Node(set.size(), x, y);
                set.add(n);
            }

        });
    }

    @Override
    public void write(Writer writer) throws IOException {
        writer.write(width + " " + height + " " + radius + "\n");
        for (NodeSet set : sets) {
            for (Node n : set) {
                if (n != set.get(0)) {
                    writer.write(" ");
                }
                writer.write(n.x() + ";" + n.y());
            }
            writer.write("\n");
        }
    }

    @Override
    public Output parseSolution(BufferedReader read) throws IOException {
        Output output = new Output(this);

        for (Graph g : output.graphs) {
            String[] setspec = read.readLine().split(" ");
            int i = 0;
            for (String pt : setspec) {
                String[] xy = pt.split(";");
                int x = Integer.parseInt(xy[0]);
                int y = Integer.parseInt(xy[1]);
                g.getVertices().get(i).set(x, y);
                i++;
            }

            String[] graphspec = read.readLine().split(" ");
            for (String ed : graphspec) {
                String[] startend = ed.split(";");
                int start = Integer.parseInt(startend[0]);
                int end = Integer.parseInt(startend[1]);
                g.addEdge(g.getVertices().get(start), g.getVertices().get(end));
            }
        }

        return output;
    }

    @Override
    public void draw(GeometryRenderer render) {

        render.setSizeMode(SizeMode.VIEW);

        // draw the grid
        render.setStroke(ExtendedColors.lightGray, 1, Dashing.SOLID);
        render.setFill(null, Hashures.SOLID);

        boolean drawgridlines = ((GeometryPanel) render).convertWorldToView(1) > 4;

        if (drawgridlines) {
            for (int x = 0; x <= width; x++) {
                render.draw(LineSegment.byStartDirectionAndLength(Vector.right(x), Vector.up(), height));
            }
            for (int y = 0; y <= height; y++) {
                render.draw(LineSegment.byStartDirectionAndLength(Vector.up(y), Vector.right(), width));
            }
        } else {
            render.draw(getBoundingBox());
        }

        // draw radius diamond
        render.setPointStyle(PointStyle.CIRCLE_WHITE, 6);
        render.pushMatrix(AffineTransform.getTranslateInstance(-radius - 1, radius));
        if (radius > 0) {
            Polygon P = new Polygon(Vector.up(radius), Vector.left(radius), Vector.down(radius), Vector.right(radius));
            render.draw(P);
            if (drawgridlines) {
                for (int i = 0; i < radius; i++) {
                    double r = radius - i;
                    render.draw(new LineSegment(new Vector(i, -r), new Vector(i, r)));
                    render.draw(new LineSegment(new Vector(-r, i), new Vector(r, i)));
                    if (i > 0) {
                        render.draw(new LineSegment(new Vector(-i, -r), new Vector(-i, r)));
                        render.draw(new LineSegment(new Vector(-r, -i), new Vector(r, -i)));
                    }
                }
            }
        }
        render.draw(Vector.origin());
        render.popMatrix();
        render.setTextStyle(TextAnchor.BOTTOM_LEFT, 18);
        render.draw(Vector.left(2 * radius + 1), "R");

        // draw the points
        render.setPointStyle(PointStyle.CIRCLE_WHITE, 6);
        for (NodeSet set : sets) {
            render.setStroke(getColor(set), 1, Dashing.SOLID);
            render.draw(set);
        }

    }

    public Color getColor(NodeSet set) {

        if (colors == null) {
            if (sets.size() <= ExtendedColors.dark.length) {
                colors = ExtendedColors.dark;
            } else if (sets.size() <= ExtendedColors.paired.length) {
                colors = ExtendedColors.paired;
            } else {
                colors = new Color[sets.size()];
                int rounds = (int) Math.ceil(sets.size() / (double) (ExtendedColors.paired.length / 2));
                int c = 0;
                int r = rounds;
                for (int i = 0; i < sets.size(); i++) {
                    int red = (int) Math.round(DoubleUtil.interpolate(ExtendedColors.paired[c].getRed(), ExtendedColors.paired[c + 1].getRed(), r / (double) rounds));
                    int green = (int) Math.round(DoubleUtil.interpolate(ExtendedColors.paired[c].getGreen(), ExtendedColors.paired[c + 1].getGreen(), r / (double) rounds));
                    int blue = (int) Math.round(DoubleUtil.interpolate(ExtendedColors.paired[c].getBlue(), ExtendedColors.paired[c + 1].getBlue(), r / (double) rounds));
                    colors[i] = new Color(red, green, blue);

                    c += 2;
                    if (c >= ExtendedColors.paired.length) {
                        c = 0;
                        r--;
                    }
                }
            }
        }

        return colors[set.id];
    }

    @Override
    public Rectangle getBoundingBox() {
        return new Rectangle(0, width, 0, height);
    }

    @Override
    public boolean isValidInstance() {
        return true;
    }

    public void sortToInput() {
        sets.sort((a, b) -> Integer.compare(a.id, b.id));
        for (NodeSet set : sets) {
            set.sort((a, b) -> Integer.compare(a.id, b.id));
        }
    }
}
