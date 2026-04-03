/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contest_2ima20.client.algorithms;

import contest_2ima20.client.schematrees.SchematicTreesAlgorithm;
import contest_2ima20.core.schematrees.Graph;
import contest_2ima20.core.schematrees.Input;
import contest_2ima20.core.schematrees.Output;
import contest_2ima20.core.schematrees.Position;
import nl.tue.geometrycore.util.IntegerUtil;

/**
 *
 * @author Wouter Meulemans (w.meulemans@tue.nl)
 */
public class ArbitraryAlgorithm extends SchematicTreesAlgorithm {

    @Override
    public Output doAlgorithm(Input input) {
        Output output = new Output(input);
        // this initializes the graphs with the necessary vertices, but not the edges

        // shift all node positions by the radius
        for (Graph g : output.graphs) {
            for (Position p : g.getVertices()) {
                p.translate(input.radius / 2, 0);
                p.setX(IntegerUtil.clipValue(p.x(), 0, input.width));
            }
        }

        // add some edges to connect
        for (Graph g : output.graphs) {
            Position prev = null;
            for (Position p : g.getVertices()) {
                if (prev != null) {
                    g.addEdge(prev, p);
                }
                prev = p;
            }
        }

        // shift all node positions by the radius
        // NB: the geometry of the edges is automatically updated
        for (Graph g : output.graphs) {
            for (Position p : g.getVertices()) {
                p.translate(0, input.radius / 2);
                p.setY(IntegerUtil.clipValue(p.y(), 0, input.width));
            }
        }

        // NB: if your algorithm changes the input lists or the output lists somehow, you may want to resort them
        // otherwise, the output format doesnt match for the input anymore
        // this algorithm doesn't, so no need!
        // input.sortToInput();
        // output.sortToInput();
        // return the result (it's going to be marvelous)
        return output;
    }

}
