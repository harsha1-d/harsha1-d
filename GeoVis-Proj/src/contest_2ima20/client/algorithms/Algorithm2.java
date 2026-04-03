package contest_2ima20.client.algorithms;

import contest_2ima20.client.schematrees.SchematicTreesAlgorithm;
import contest_2ima20.core.schematrees.Graph;
import contest_2ima20.core.schematrees.Input;
import contest_2ima20.core.schematrees.Output;
import contest_2ima20.core.schematrees.Position;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import nl.tue.geometrycore.algorithms.mst.SimpleEuclideanMST;
import nl.tue.geometrycore.util.Pair;

public class Algorithm2 extends SchematicTreesAlgorithm {

    private static final int MAX_REFINEMENT_PASSES = 3;
    private static final int MAX_POSITIONS_PER_GRAPH = 24;
    private static final int MAX_CANDIDATES_PER_POSITION = 96;
    private static final long REFINEMENT_TIME_BUDGET_NANOS = 8_000_000_000L;

    public String getName() {
        return "Global Matching + Refinement";
    }

    @Override
    public Output doAlgorithm(Input input) {
        Output output = new Output(input);

        List<VertexRecord> vertices = collectVertices(output);
        assignUniquePositions(input, vertices);

        for (Graph graph : output.graphs) {
            rebuildGraph(graph);
        }

        improveCrossings(output);

        return output;
    }

    private List<VertexRecord> collectVertices(Output output) {
        List<VertexRecord> vertices = new ArrayList<>();
        for (Graph graph : output.graphs) {
            for (Position position : graph.getVertices()) {
                vertices.add(new VertexRecord(position));
            }
        }
        return vertices;
    }

    private void assignUniquePositions(Input input, List<VertexRecord> vertices) {
        Map<Long, Integer> cellIndexByKey = new HashMap<>();
        List<GridCell> cells = new ArrayList<>();

        for (VertexRecord vertex : vertices) {
            enumerateCandidates(input, vertex, cellIndexByKey, cells);
        }

        vertices.sort(Comparator
                .comparingInt((VertexRecord vertex) -> vertex.candidateCells.length)
                .thenComparingInt(vertex -> -vertex.totalDemand)
                .thenComparingInt(vertex -> vertex.position.node.id));

        int vertexCount = vertices.size();
        int cellCount = cells.size();
        int[] cellToVertex = new int[cellCount];
        int[] vertexToCell = new int[vertexCount];
        int[] seenVertices = new int[vertexCount];
        int[] seenCells = new int[cellCount];
        Arrays.fill(cellToVertex, -1);
        Arrays.fill(vertexToCell, -1);

        for (int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
            if (!tryAssign(vertexIndex, vertexToCell, cellToVertex, seenVertices, seenCells, vertices, vertexIndex + 1)) {
                throw new IllegalStateException("Could not find a valid unique placement for all vertices.");
            }
        }

        for (int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
            GridCell cell = cells.get(vertexToCell[vertexIndex]);
            vertices.get(vertexIndex).position.set(cell.x, cell.y);
        }
    }

    private void enumerateCandidates(
            Input input,
            VertexRecord vertex,
            Map<Long, Integer> cellIndexByKey,
            List<GridCell> cells
    ) {
        int radius = input.radius;
        int baseX = vertex.position.node.x();
        int baseY = vertex.position.node.y();

        List<Integer> candidateIds = new ArrayList<>(1 + 2 * radius * (radius + 1));
        for (int dx = -radius; dx <= radius; dx++) {
            int remaining = radius - Math.abs(dx);
            for (int dy = -remaining; dy <= remaining; dy++) {
                int x = baseX + dx;
                int y = baseY + dy;

                if (x < 0 || x > input.width || y < 0 || y > input.height) {
                    continue;
                }

                long key = key(x, y);
                Integer cellIndex = cellIndexByKey.get(key);
                if (cellIndex == null) {
                    cellIndex = cells.size();
                    cellIndexByKey.put(key, cellIndex);
                    cells.add(new GridCell(x, y));
                }

                candidateIds.add(cellIndex);
            }
        }

        for (int candidateId : candidateIds) {
            cells.get(candidateId).demand++;
        }

        candidateIds.sort((leftId, rightId) -> compareCells(cells.get(leftId), cells.get(rightId), baseX, baseY));

        vertex.candidateCells = new int[candidateIds.size()];
        int demand = 0;
        for (int i = 0; i < candidateIds.size(); i++) {
            int candidateId = candidateIds.get(i);
            vertex.candidateCells[i] = candidateId;
            demand += cells.get(candidateId).demand;
        }
        vertex.totalDemand = demand;
    }

    private int compareCells(GridCell left, GridCell right, int baseX, int baseY) {
        int leftDistance = Math.abs(left.x - baseX) + Math.abs(left.y - baseY);
        int rightDistance = Math.abs(right.x - baseX) + Math.abs(right.y - baseY);
        if (leftDistance != rightDistance) {
            return Integer.compare(leftDistance, rightDistance);
        }
        if (left.demand != right.demand) {
            return Integer.compare(left.demand, right.demand);
        }
        if (left.x != right.x) {
            return Integer.compare(left.x, right.x);
        }
        return Integer.compare(left.y, right.y);
    }

    private boolean tryAssign(
            int vertexIndex,
            int[] vertexToCell,
            int[] cellToVertex,
            int[] seenVertices,
            int[] seenCells,
            List<VertexRecord> vertices,
            int visitToken
    ) {
        if (seenVertices[vertexIndex] == visitToken) {
            return false;
        }
        seenVertices[vertexIndex] = visitToken;

        for (int cellIndex : vertices.get(vertexIndex).candidateCells) {
            if (seenCells[cellIndex] == visitToken) {
                continue;
            }
            seenCells[cellIndex] = visitToken;

            int occupant = cellToVertex[cellIndex];
            if (occupant == -1 || tryAssign(occupant, vertexToCell, cellToVertex, seenVertices, seenCells, vertices, visitToken)) {
                cellToVertex[cellIndex] = vertexIndex;
                vertexToCell[vertexIndex] = cellIndex;
                return true;
            }
        }

        return false;
    }

    private void rebuildGraph(Graph graph) {
        graph.clearEdges();

        if (graph.getVertices().size() < 2) {
            return;
        }

        SimpleEuclideanMST<Position> mst = new SimpleEuclideanMST<>();
        mst.run(graph.getVertices());
        for (Pair<Position, Position> edge : mst.edges()) {
            graph.addEdge(edge.getFirst(), edge.getSecond());
        }
    }

    private void improveCrossings(Output output) {
        List<Graph> graphs = output.graphs;
        Set<Long> occupied = buildOccupiedSet(output);
        Map<Position, List<GridPoint>> candidateCache = new HashMap<>();
        double bestScore = output.computeQuality();
        long deadline = System.nanoTime() + REFINEMENT_TIME_BUDGET_NANOS;

        for (int pass = 0; pass < MAX_REFINEMENT_PASSES; pass++) {
            if (System.nanoTime() >= deadline || bestScore == 0) {
                return;
            }

            boolean improved = false;

            for (Graph graph : graphs) {
                if (System.nanoTime() >= deadline) {
                    return;
                }

                CrossingStats stats = analyzeGraphCrossings(graph, graphs);
                if (stats.crossingCount == 0) {
                    continue;
                }

                List<Position> positions = new ArrayList<>(stats.involvedVertices);
                positions.sort((a, b) -> Integer.compare(
                        stats.vertexCrossings.getOrDefault(b, 0),
                        stats.vertexCrossings.getOrDefault(a, 0)));

                int positionLimit = Math.min(MAX_POSITIONS_PER_GRAPH, positions.size());
                for (int i = 0; i < positionLimit; i++) {
                    if (System.nanoTime() >= deadline) {
                        return;
                    }

                    Position position = positions.get(i);
                    int originalX = position.x();
                    int originalY = position.y();
                    long originalKey = key(originalX, originalY);
                    int currentGraphScore = countGraphIntersections(graph, graphs);

                    occupied.remove(originalKey);

                    int bestX = originalX;
                    int bestY = originalY;
                    long bestKey = originalKey;
                    int bestGraphScore = currentGraphScore;

                    List<GridPoint> candidates = enumerateCandidates(output.input, position, candidateCache);
                    int candidateLimit = Math.min(MAX_CANDIDATES_PER_POSITION, candidates.size());
                    for (int c = 0; c < candidateLimit; c++) {
                        GridPoint candidate = candidates.get(c);
                        long candidateKey = key(candidate.x, candidate.y);
                        if (candidateKey == originalKey || occupied.contains(candidateKey)) {
                            continue;
                        }

                        position.set(candidate.x, candidate.y);
                        rebuildGraph(graph);

                        int candidateGraphScore = countGraphIntersections(graph, graphs);
                        if (candidateGraphScore < bestGraphScore) {
                            bestGraphScore = candidateGraphScore;
                            bestX = candidate.x;
                            bestY = candidate.y;
                            bestKey = candidateKey;
                        }
                    }

                    if (bestGraphScore < currentGraphScore) {
                        position.set(bestX, bestY);
                        rebuildGraph(graph);
                        occupied.add(bestKey);
                        bestScore = bestScore - currentGraphScore + bestGraphScore;
                        improved = true;
                    } else {
                        position.set(originalX, originalY);
                        rebuildGraph(graph);
                        occupied.add(originalKey);
                    }
                }
            }

            if (!improved) {
                return;
            }
        }
    }

    private Set<Long> buildOccupiedSet(Output output) {
        Set<Long> occupied = new HashSet<>();
        for (Graph graph : output.graphs) {
            for (Position position : graph.getVertices()) {
                occupied.add(key(position.x(), position.y()));
            }
        }
        return occupied;
    }

    private List<GridPoint> enumerateCandidates(Input input, Position position, Map<Position, List<GridPoint>> candidateCache) {
        List<GridPoint> cached = candidateCache.get(position);
        if (cached != null) {
            return cached;
        }

        int radius = input.radius;
        int baseX = position.node.x();
        int baseY = position.node.y();
        List<GridPoint> candidates = new ArrayList<>(1 + 2 * radius * (radius + 1));

        for (int dx = -radius; dx <= radius; dx++) {
            int remaining = radius - Math.abs(dx);
            for (int dy = -remaining; dy <= remaining; dy++) {
                int x = baseX + dx;
                int y = baseY + dy;

                if (x < 0 || x > input.width || y < 0 || y > input.height) {
                    continue;
                }

                candidates.add(new GridPoint(x, y, Math.abs(dx) + Math.abs(dy)));
            }
        }

        candidates.sort((a, b) -> Integer.compare(a.distance, b.distance));
        candidateCache.put(position, candidates);
        return candidates;
    }

    private int countGraphIntersections(Graph graph, List<Graph> graphs) {
        int count = 0;
        List<contest_2ima20.core.schematrees.Edge> graphEdges = graph.getEdges();

        for (int i = 0; i < graphEdges.size(); i++) {
            contest_2ima20.core.schematrees.Edge edge = graphEdges.get(i);

            for (Graph otherGraph : graphs) {
                List<contest_2ima20.core.schematrees.Edge> otherEdges = otherGraph.getEdges();
                int startIndex = otherGraph == graph ? i + 1 : 0;

                for (int j = startIndex; j < otherEdges.size(); j++) {
                    contest_2ima20.core.schematrees.Edge other = otherEdges.get(j);
                    if (edge.getCommonVertex(other) != null) {
                        continue;
                    }
                    if (!edge.getGeometry().intersect(other.getGeometry()).isEmpty()) {
                        count++;
                    }
                }
            }
        }

        return count;
    }

    private CrossingStats analyzeGraphCrossings(Graph graph, List<Graph> graphs) {
        int count = 0;
        Set<Position> involvedVertices = new HashSet<>();
        Map<Position, Integer> vertexCrossings = new HashMap<>();
        List<contest_2ima20.core.schematrees.Edge> graphEdges = graph.getEdges();

        for (int i = 0; i < graphEdges.size(); i++) {
            contest_2ima20.core.schematrees.Edge edge = graphEdges.get(i);

            for (Graph otherGraph : graphs) {
                List<contest_2ima20.core.schematrees.Edge> otherEdges = otherGraph.getEdges();
                int startIndex = otherGraph == graph ? i + 1 : 0;

                for (int j = startIndex; j < otherEdges.size(); j++) {
                    contest_2ima20.core.schematrees.Edge other = otherEdges.get(j);
                    if (edge.getCommonVertex(other) != null) {
                        continue;
                    }
                    if (!edge.getGeometry().intersect(other.getGeometry()).isEmpty()) {
                        count++;
                        involvedVertices.add(edge.getStart());
                        involvedVertices.add(edge.getEnd());
                        incrementCrossing(vertexCrossings, edge.getStart());
                        incrementCrossing(vertexCrossings, edge.getEnd());
                    }
                }
            }
        }

        return new CrossingStats(count, involvedVertices, vertexCrossings);
    }

    private void incrementCrossing(Map<Position, Integer> vertexCrossings, Position position) {
        vertexCrossings.put(position, vertexCrossings.getOrDefault(position, 0) + 1);
    }

    private long key(int x, int y) {
        return (((long) x) << 32) ^ (y & 0xffffffffL);
    }

    private static class VertexRecord {

        final Position position;
        int[] candidateCells;
        int totalDemand;

        VertexRecord(Position position) {
            this.position = position;
        }
    }

    private static class GridCell {

        final int x;
        final int y;
        int demand;

        GridCell(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    private static class GridPoint {

        final int x;
        final int y;
        final int distance;

        GridPoint(int x, int y, int distance) {
            this.x = x;
            this.y = y;
            this.distance = distance;
        }
    }

    private static class CrossingStats {

        final int crossingCount;
        final Set<Position> involvedVertices;
        final Map<Position, Integer> vertexCrossings;

        CrossingStats(int crossingCount, Set<Position> involvedVertices, Map<Position, Integer> vertexCrossings) {
            this.crossingCount = crossingCount;
            this.involvedVertices = involvedVertices;
            this.vertexCrossings = vertexCrossings;
        }
    }
}
