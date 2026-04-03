package contest_2ima20.core.problem;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

public abstract class Problem<TOutput extends Solution> implements Viewable {

    private String instanceName;
    private HashMap<String, TOutput> solutions;
    private File instanceFile;
    private boolean isLoaded;

    public String instanceName() {
        return instanceName;
    }

    public abstract boolean isValidInstance();

    /**
     * File that should contain a problem definition
     *
     * @param f
     */
    public final void initializeProblem(File f) {
        try {
            instanceFile = f;
            instanceName = f.getName().substring(0, f.getName().length() - 4);

            read(new BufferedReader(new FileReader(f)));
        } catch (IOException e) {
            System.err.println("An " + e.getClass().getSimpleName() + " occurred while loading the file " + f.getName());
            e.printStackTrace();
        }

    }

    public final boolean loadProblem() {
        if (!isLoaded) {
            try {
                read(new BufferedReader(new FileReader(instanceFile)));
                isLoaded = true;
            } catch (IOException e) {
                System.err.println("An " + e.getClass().getSimpleName() + " occurred while loading the file " + instanceFile.getName());
                e.printStackTrace();
            }
        }
        return isLoaded;
    }

    public abstract void read(BufferedReader read) throws IOException;

    public abstract void write(Writer out) throws IOException;

    public boolean write(File exportFile) {
        try (BufferedWriter out = new BufferedWriter(new FileWriter(exportFile))) {

            write(out);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    public String write() {
        try {
            StringWriter build = new StringWriter();
            write(build);
            return build.toString();
        } catch (IOException ex) {
            Logger.getLogger(Solution.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
    }

    /**
     * Load solution from file
     *
     * @param f
     * @return
     */
    public final TOutput loadSolution(String name, File f) {
        TOutput s = parseSolution(f);
        if (s != null) {
            s.setName(name);
            putSolution(name, s);
        }
        return s;
    }

    public TOutput parseSolution(File file) {
        try (BufferedReader read = new BufferedReader(new FileReader(file))) {
            return parseSolution(read);
        } catch (IOException ex) {
            Logger.getLogger(Problem.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
    }

    public TOutput parseSolution(String solution) {
        try {
            return parseSolution(new BufferedReader(new StringReader(solution)));
        } catch (IOException ex) {
            Logger.getLogger(Problem.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
    }

    public abstract TOutput parseSolution(BufferedReader read) throws IOException;

    /**
     * adds or updates a solution in the set
     *
     * @param name identifier for the solution
     * @param solution representation of the solution
     */
    public void putSolution(String name, TOutput solution) {
        solutions = solutions == null ? new HashMap<>() : solutions;
        if (name != null && !name.isEmpty()) {
            solutions.put(name, solution);
        }
    }

    /**
     * Removes a solution from the set of solutions
     *
     * @param name identifier of the solution
     */
    public void removeSolution(String name) {
        solutions = solutions == null ? new HashMap<>() : solutions;
        if (name != null && !name.isEmpty()) {
            solutions.remove(name);
        }
    }

    /**
     * Get solution known by a specified name. If not known null will be
     * returned
     *
     * @param name identifier of the solution
     * @return Solution format
     */
    public Solution getSolution(String name) {
        solutions = solutions == null ? new HashMap<>() : solutions;
        try {
            if (name != null && !name.isEmpty()) {
                return solutions.get(name);
            }
        } catch (ClassCastException e) {
        }
        return null;
    }

    public Collection<TOutput> getAllSolutions() {
        if (solutions != null) {
            return solutions.values();
        } else {
            return new ArrayList<>();
        }
    }
}
