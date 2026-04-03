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
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Algorithm1 extends SchematicTreesAlgorithm {

    private static final int MAX_PASSES = 3;
    private static final int LARGE_MAX_PASSES = 2;
    private static final int LARGE_MAX_POSITIONS_PER_GRAPH = 20;
    private static final int LARGE_MAX_CANDIDATES_PER_POSITION = 128;
    private static final long LARGE_TIME_BUDGET_NANOS = 15_000_000_000L;
    private static final long LARGE_INSTANCE_WORK_THRESHOLD = 100_000_000L;

    public String getName() {
        return "MST + Local Search";
    }

    @Override
    public Output doAlgorithm(Input input) {
        String infeasibilityReason = findImmediateInfeasibility(input);
        if (infeasibilityReason != null) {
            System.err.println("Algorithm1 detected infeasible instance: " + infeasibilityReason);
            return buildInvalidOutput(input);
        }

        try {
            Output output = new Output(input);
            assignUniquePositions(input, output);

            for (Graph graph : output.graphs) {
                rebuildGraph(graph);
            }

            improvePositions(output);
            return output;
        } catch (RuntimeException exception) {
            System.err.println("Algorithm1 failed, falling back to Algorithm2: " + exception.getMessage());
            exception.printStackTrace();
            return new Algorithm2().doAlgorithm(input);
        }
    }

    private String findImmediateInfeasibility(Input input) {
        Map<Long, Integer> multiplicityByOrigin = new HashMap<>();
        for (contest_2ima20.core.schematrees.NodeSet set : input.sets) {
            for (contest_2ima20.core.schematrees.Node node : set) {
                long originKey = key(node.x(), node.y());
                multiplicityByOrigin.put(originKey, multiplicityByOrigin.getOrDefault(originKey, 0) + 1);
            }
        }

        for (contest_2ima20.core.schematrees.NodeSet set : input.sets) {
            for (contest_2ima20.core.schematrees.Node node : set) {
                long originKey = key(node.x(), node.y());
                Integer multiplicity = multiplicityByOrigin.remove(originKey);
                if (multiplicity == null) {
                    continue;
                }

                int candidateCount = enumerateCandidates(input, node.x(), node.y()).size();
                if (multiplicity > candidateCount) {
                    return "origin (" + node.x() + "," + node.y() + ") needs "
                            + multiplicity + " unique cells but only has " + candidateCount;
                }
            }
        }

        return null;
    }

    private Output buildInvalidOutput(Input input) {
        Output output = new Output(input);
        for (Graph graph : output.graphs) {
            rebuildGraph(graph);
        }
        return output;
    }

    private void assignUniquePositions(Input input, Output output) {
        List<VertexRecord> vertices = new ArrayList<>();
        for (Graph graph : output.graphs) {
            for (Position position : graph.getVertices()) {
                vertices.add(new VertexRecord(position));
            }
        }

        Map<Long, Integer> cellIndexByKey = new HashMap<>();
        List<GridCell> cells = new ArrayList<>();
        for (VertexRecord vertex : vertices) {
            enumerateCandidateCells(input, vertex, cellIndexByKey, cells);
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
                throw new IllegalStateException("Could not find a valid unique placement for all duplicated vertices.");
            }
        }

        for (int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
            GridCell cell = cells.get(vertexToCell[vertexIndex]);
            vertices.get(vertexIndex).position.set(cell.x, cell.y);
        }
    }

    private void enumerateCandidateCells(
            Input input,
            VertexRecord vertex,
            Map<Long, Integer> cellIndexByKey,
            List<GridCell> cells
    ) {
        int baseX = vertex.position.node.x();
        int baseY = vertex.position.node.y();
        List<GridPoint> candidates = enumerateCandidates(input, baseX, baseY);
        List<Integer> candidateIds = new ArrayList<>(candidates.size());

        for (GridPoint candidate : candidates) {
            long candidateKey = key(candidate.x, candidate.y);
            Integer cellIndex = cellIndexByKey.get(candidateKey);
            if (cellIndex == null) {
                cellIndex = cells.size();
                cellIndexByKey.put(candidateKey, cellIndex);
                cells.add(new GridCell(candidate.x, candidate.y));
            }
            candidateIds.add(cellIndex);
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

    private void improvePositions(Output output) {
        if (isLargeInstance(output.input)) {
            improvePositionsLarge(output);
            return;
        }

        List<Graph> graphs = output.graphs;
        Set<Long> occupied = buildOccupiedSet(output);
        Map<Position, List<GridPoint>> candidateCache = new HashMap<>();
        double bestScore = output.computeQuality();
        int totalGraphs = graphs.size();

        for (int pass = 0; pass < MAX_PASSES; pass++) {
            boolean improved = false;

            for (int graphIndex = 0; graphIndex < totalGraphs; graphIndex++) {
                Graph graph = graphs.get(graphIndex);
                for (Position position : graph.getVertices()) {
                    int originalX = position.x();
                    int originalY = position.y();
                    long originalKey = key(originalX, originalY);
                    double currentGraphScore = countGraphIntersections(graph, output);

                    occupied.remove(originalKey);

                    List<GridPoint> candidates = enumerateCandidates(output.input, position, candidateCache);
                    for (GridPoint candidate : candidates) {
                        long candidateKey = key(candidate.x, candidate.y);
                        if (occupied.contains(candidateKey)) {
                            continue;
                        }
                        if (candidate.x == originalX && candidate.y == originalY) {
                            continue;
                        }

                        position.set(candidate.x, candidate.y);
                        rebuildGraph(graph);

                        double candidateGraphScore = countGraphIntersections(graph, output);
                        double candidateScore = bestScore - currentGraphScore + candidateGraphScore;
                        if (candidateScore < bestScore) {
                            bestScore = candidateScore;
                            currentGraphScore = candidateGraphScore;
                            occupied.add(candidateKey);
                            improved = true;
                            originalX = candidate.x;
                            originalY = candidate.y;
                            originalKey = candidateKey;
                            break;
                        }
                    }

                    if (!occupied.contains(originalKey)) {
                        position.set(originalX, originalY);
                        rebuildGraph(graph);
                        occupied.add(originalKey);
                    }
                }
            }

            if (!improved || bestScore == 0) {
                break;
            }
        }
    }

    private void improvePositionsLarge(Output output) {

        List<Graph> graphs = output.graphs;
        Set<Long> occupied = buildOccupiedSet(output);
        Map<Position, List<GridPoint>> candidateCache = new HashMap<>();
        double bestScore = output.computeQuality();
        long deadline = System.nanoTime() + LARGE_TIME_BUDGET_NANOS;
        System.out.println("Algorithm1 large-instance search started with score " + bestScore);

        for (int pass = 0; pass < LARGE_MAX_PASSES; pass++) {
            if (System.nanoTime() >= deadline) {
                System.out.println("Algorithm1 large-instance search stopped before pass "
                        + (pass + 1) + " due to time budget");
                break;
            }

            boolean improved = false;
            System.out.println("Algorithm1 large-instance pass " + (pass + 1) + "/" + LARGE_MAX_PASSES
                    + " started, current best score " + bestScore);

            for (int graphIndex = 0; graphIndex < graphs.size(); graphIndex++) {
                Graph graph = graphs.get(graphIndex);
                if (System.nanoTime() >= deadline) {
                    System.out.println("Algorithm1 large-instance search stopped during pass "
                            + (pass + 1) + " due to time budget");
                    break;
                }

                CrossingStats stats = analyzeGraphCrossings(graph, graphs);
                if (stats.crossingCount == 0) {
                    System.out.println("Algorithm1 pass " + (pass + 1)
                            + ": graph " + (graphIndex + 1) + "/" + graphs.size()
                            + " has no crossings, skipping");
                    continue;
                }

                List<Position> positions = new ArrayList<>(stats.involvedVertices);
                Map<Position, Integer> vertexCrossings = stats.vertexCrossings;
                positions.sort((a, b) -> Integer.compare(
                        vertexCrossings.getOrDefault(b, 0),
                        vertexCrossings.getOrDefault(a, 0)));

                int positionLimit = Math.min(LARGE_MAX_POSITIONS_PER_GRAPH, positions.size());
                System.out.println("Algorithm1 pass " + (pass + 1)
                        + ": graph " + (graphIndex + 1) + "/" + graphs.size()
                        + " checking " + positionLimit + " positions out of " + positions.size()
                        + ", graph crossings " + stats.crossingCount);
                for (int i = 0; i < positionLimit; i++) {
                    if (System.nanoTime() >= deadline) {
                        break;
                    }

                    Position position = positions.get(i);
                    int originalX = position.x();
                    int originalY = position.y();
                    long originalKey = key(originalX, originalY);
                    int currentGraphScore = stats.crossingCount;
                    System.out.println("Algorithm1 pass " + (pass + 1)
                            + ": graph " + (graphIndex + 1) + "/" + graphs.size()
                            + ", position " + (i + 1) + "/" + positionLimit
                            + " at (" + originalX + "," + originalY + ")"
                            + ", graph crossings " + currentGraphScore
                            + ", best score " + bestScore);

                    occupied.remove(originalKey);

                    List<GridPoint> candidates = enumerateCandidates(output.input, position, candidateCache);
                    int candidateLimit = Math.min(LARGE_MAX_CANDIDATES_PER_POSITION, candidates.size());
                    int bestX = originalX;
                    int bestY = originalY;
                    long bestKey = originalKey;
                    int bestGraphScore = currentGraphScore;

                    for (int c = 0; c < candidateLimit; c++) {
                        GridPoint candidate = candidates.get(c);
                        long candidateKey = key(candidate.x, candidate.y);
                        if (occupied.contains(candidateKey)) {
                            continue;
                        }
                        if (candidate.x == originalX && candidate.y == originalY) {
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
                        bestScore = bestScore - currentGraphScore + bestGraphScore;
                        occupied.add(bestKey);
                        improved = true;
                        System.out.println("Algorithm1 improvement: pass " + (pass + 1)
                                + ", graph " + (graphIndex + 1) + "/" + graphs.size()
                                + ", position " + (i + 1) + "/" + positionLimit
                                + " moved to (" + bestX + "," + bestY + ")"
                                + ", graph crossings " + currentGraphScore + " -> " + bestGraphScore
                                + ", best score now " + bestScore);
                        stats = analyzeGraphCrossings(graph, graphs);
                    } else {
                        position.set(originalX, originalY);
                        rebuildGraph(graph);
                        occupied.add(originalKey);
                    }

                    if (bestScore == 0) {
                        System.out.println("Algorithm1 large-instance search reached score 0");
                        return;
                    }
                }
            }

            if (!improved) {
                System.out.println("Algorithm1 large-instance pass " + (pass + 1)
                        + " finished without improvement");
                break;
            }
        }

        System.out.println("Algorithm1 large-instance search finished with score " + bestScore);
    }

    private boolean isLargeInstance(Input input) {
        int totalVertices = 0;
        int totalEdgeCount = 0;
        for (contest_2ima20.core.schematrees.NodeSet set : input.sets) {
            totalVertices += set.size();
            totalEdgeCount += Math.max(0, set.size() - 1);
        }
        int candidatesPerVertex = 1 + 2 * input.radius * (input.radius + 1);
        long estimatedWork = (long) totalVertices * candidatesPerVertex * Math.max(1, totalEdgeCount);
        return estimatedWork > LARGE_INSTANCE_WORK_THRESHOLD;
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

        List<GridPoint> candidates = enumerateCandidates(input, position.node.x(), position.node.y());
        candidateCache.put(position, candidates);
        return candidates;
    }

    private List<GridPoint> enumerateCandidates(Input input, int baseX, int baseY) {
        int radius = input.radius;
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
        return candidates;
    }

    private void rebuildGraph(Graph graph) {

        graph.clearEdges();

        if (graph.getVertices().size() < 2) {
            return;
        }

        List<TempEdge> edges = buildPrimMST(graph.getVertices());
        for (TempEdge edge : edges) {
            graph.addEdge(edge.u, edge.v);
        }
    }

    private List<TempEdge> buildPrimMST(List<Position> vertices) {
        int n = vertices.size();
        boolean[] inTree = new boolean[n];
        int[] parent = new int[n];
        double[] bestDistance = new double[n];

        inTree[0] = true;
        parent[0] = -1;

        for (int i = 1; i < n; i++) {
            parent[i] = 0;
            bestDistance[i] = squaredDistance(vertices.get(0), vertices.get(i));
        }

        List<TempEdge> result = new ArrayList<>(Math.max(0, n - 1));

        for (int edgeCount = 1; edgeCount < n; edgeCount++) {
            int next = -1;
            double best = Double.POSITIVE_INFINITY;

            for (int i = 0; i < n; i++) {
                if (!inTree[i] && bestDistance[i] < best) {
                    best = bestDistance[i];
                    next = i;
                }
            }

            if (next < 0) {
                break;
            }

            inTree[next] = true;
            result.add(new TempEdge(vertices.get(parent[next]), vertices.get(next)));

            for (int i = 0; i < n; i++) {
                if (inTree[i]) {
                    continue;
                }

                double candidate = squaredDistance(vertices.get(next), vertices.get(i));
                if (candidate < bestDistance[i]) {
                    bestDistance[i] = candidate;
                    parent[i] = next;
                }
            }
        }

        return result;
    }

    private double squaredDistance(Position a, Position b) {
        double dx = a.getX() - b.getX();
        double dy = a.getY() - b.getY();
        return dx * dx + dy * dy;
    }

    private int countGraphIntersections(Graph graph, Output output) {
        return countGraphIntersections(graph, output.graphs);
    }

    private int countGraphIntersections(Graph graph, List<Graph> graphs) {
        return analyzeGraphCrossings(graph, graphs, false).crossingCount;
    }

    private CrossingStats analyzeGraphCrossings(Graph graph, List<Graph> graphs) {
        return analyzeGraphCrossings(graph, graphs, true);
    }

    private CrossingStats analyzeGraphCrossings(Graph graph, List<Graph> graphs, boolean collectVertexStats) {
        Map<contest_2ima20.core.schematrees.Edge, Boolean> targetEdges = new IdentityHashMap<>();
        Map<contest_2ima20.core.schematrees.Edge, Integer> targetEdgeOrder = new IdentityHashMap<>();
        List<SweepEvent> events = new ArrayList<>();

        int edgeOrder = 0;
        for (Graph currentGraph : graphs) {
            for (contest_2ima20.core.schematrees.Edge edge : currentGraph.getEdges()) {
                boolean target = currentGraph == graph;
                if (target) {
                    targetEdges.put(edge, Boolean.TRUE);
                    targetEdgeOrder.put(edge, edgeOrder++);
                }

                Position left;
                Position right;
                if (edge.getStart().x() <= edge.getEnd().x()) {
                    left = edge.getStart();
                    right = edge.getEnd();
                } else {
                    left = edge.getEnd();
                    right = edge.getStart();
                }

                events.add(new SweepEvent(edge, target, true, 2L * left.x()));
                events.add(new SweepEvent(edge, target, false, 2L * right.x() + 1));
            }
        }

        events.sort((a, b) -> Long.compare(a.x, b.x));

        int count = 0;
        Set<Position> involvedVertices = collectVertexStats ? new HashSet<>() : null;
        Map<Position, Integer> vertexCrossings = collectVertexStats ? new HashMap<>() : null;
        List<contest_2ima20.core.schematrees.Edge> active = new ArrayList<>();

        for (SweepEvent event : events) {
            if (event.start) {
                for (contest_2ima20.core.schematrees.Edge other : active) {
                    boolean otherIsTarget = targetEdges.containsKey(other);
                    if (event.target == otherIsTarget) {
                        if (!event.target) {
                            continue;
                        }
                    }

                    if (event.edge.getCommonVertex(other) != null) {
                        continue;
                    }
                    if (event.edge.getGeometry().intersect(other.getGeometry()).isEmpty()) {
                        continue;
                    }

                    count++;
                    if (collectVertexStats) {
                        contest_2ima20.core.schematrees.Edge countedTargetEdge;
                        if (event.target && otherIsTarget) {
                            countedTargetEdge = targetEdgeOrder.get(event.edge) < targetEdgeOrder.get(other)
                                    ? event.edge
                                    : other;
                        } else {
                            countedTargetEdge = event.target ? event.edge : other;
                        }

                        involvedVertices.add(countedTargetEdge.getStart());
                        involvedVertices.add(countedTargetEdge.getEnd());
                        incrementCrossing(vertexCrossings, countedTargetEdge.getStart());
                        incrementCrossing(vertexCrossings, countedTargetEdge.getEnd());
                    }
                }

                active.add(event.edge);
            } else {
                active.remove(event.edge);
            }
        }

        return new CrossingStats(count,
                collectVertexStats ? involvedVertices : new HashSet<>(),
                collectVertexStats ? vertexCrossings : new HashMap<>());
    }

    private void incrementCrossing(Map<Position, Integer> vertexCrossings, Position position) {
        vertexCrossings.put(position, vertexCrossings.getOrDefault(position, 0) + 1);
    }

    private long key(int x, int y) {
        return (((long) x) << 32) ^ (y & 0xffffffffL);
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

    private static class TempEdge {

        final Position u;
        final Position v;

        TempEdge(Position u, Position v) {
            this.u = u;
            this.v = v;
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

    private static class SweepEvent {

        final contest_2ima20.core.schematrees.Edge edge;
        final boolean target;
        final boolean start;
        final long x;

        SweepEvent(contest_2ima20.core.schematrees.Edge edge, boolean target, boolean start, long x) {
            this.edge = edge;
            this.target = target;
            this.start = start;
            this.x = x;
        }
    }

}
