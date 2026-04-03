package contest_2ima20.client.gui;

import contest_2ima20.core.util.Settings;
import contest_2ima20.core.problem.Problem;

import javax.swing.*;
import javax.swing.border.BevelBorder;
import javax.swing.border.MatteBorder;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class ProblemsPane extends JPanel {

    private File directory;
    private File oldDirectory;
    private Problem selectedProblem;
    private static Class[] problemTypes;
    private static ArrayList<Problem> problems;
    private static HashMap<String, Problem> problemSet;
    private static ArrayList<String> problemNames;
    private ArrayList<Long> editTimes;
    private ArrayList<ProblemPane> visualizations;
    private ArrayList<ProblemListener> listeners;
    private MainPane parent;
    private final Object problemLock = new Object();

    public ProblemsPane(MainPane parent, File problemDirectory) {
        super();
        this.parent = parent;

        listeners = new ArrayList<>();
        problems = new ArrayList<>();
        problemNames = new ArrayList<>();
        problemSet = new HashMap<>();
        editTimes = new ArrayList<>();
        visualizations = new ArrayList<>();

        setBorder(new MatteBorder(3, 0, 0, 0, Color.darkGray));
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        setDirectory(problemDirectory);

        if (parent != null) {
            Thread checker = new Thread(new Runnable() {
                @Override
                public void run() {
                    while (true) {
                        synchronized (problemLock) {
                            checkProblemDirectory();
                        }
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException e) {
                        }
                    }
                }
            });

            checker.start();
        } else {
            // headless mode
            synchronized (problemLock) {
                checkProblemDirectory();
            }
        }
    }

    public void addListener(ProblemListener l) {
        if (l != null) {
            listeners.add(l);
        }
    }

    public void removeListener(ProblemListener l) {
        if (l != null) {
            listeners.remove(l);
        }
    }

    public void checkProblemDirectory() {
        if (directory != oldDirectory) {
            for (ProblemPane pane : visualizations) {
                remove(pane);
            }
            visualizations = new ArrayList<>();
            problemNames = new ArrayList<>();
            editTimes = new ArrayList<>();
            problems = new ArrayList<>();
            problemSet = new HashMap<>();
            loadProblems();
            if (SettingsPane.getExportPath() != null && parent != null && parent.getSettingsPanel() != null && SettingsPane.getExportPath().isDirectory()) {
                parent.getSettingsPanel().batchReader(SettingsPane.getExportPath());
            }
        } else {
            checkProblems();
        }
        oldDirectory = directory;
        reloadGraphics();
    }

    public void loadProblems() {
        if (directory != null && directory.isDirectory()) {
            File[] allFiles = directory.listFiles();
            Arrays.sort(allFiles, (p, q) -> {
                String pn = p.getName().replace(".txt", "");
                String qn = q.getName().replace(".txt", "");
                int pi, qi;
                try {
                    pi = Integer.parseInt(pn);
                } catch (Exception ex) {
                    pi = -1;
                }
                try {
                    qi = Integer.parseInt(qn);
                } catch (Exception ex) {
                    qi = -1;
                }
                if (qi >= 0 || pi >= 0) {
                    return Integer.compare(pi, qi);
                } else {
                    return pn.compareTo(qn);
                }
            });
            problems = new ArrayList<>();
            problemNames = new ArrayList<>();
            editTimes = new ArrayList<>();
            visualizations = new ArrayList<>();
            problemSet = new HashMap<>();

            System.out.println("Loading problems...");
            int i = 0;
            for (File file : allFiles) {
                if (!file.isDirectory()) {
                    addNewProblem(file);
                    i++;
                }
            }
            reloadGraphics();
            System.out.println("Loaded " + i + " problems");
        }
    }

    public void checkProblems() {
        if (directory != null && directory.isDirectory()) {
            File[] allFiles = directory.listFiles();
            for (File file : allFiles) {
                if (problemNames.contains(file.getName())) {
                    int index = problemNames.indexOf(file.getName());
                    if (editTimes.get(index) != file.lastModified()) {
                        Problem updatedProblem = Settings.definition.createInputInstance();
                        updatedProblem.initializeProblem(file);
                        updatedProblem.loadProblem();
                        if (updatedProblem.isValidInstance()) {
                            problems.set(index, updatedProblem);
                            editTimes.set(index, file.lastModified());
                        } else {
                            problems.remove(index);
                            editTimes.remove(index);
                            problemNames.remove(index);
                            remove(visualizations.get(index));
                            visualizations.remove(index);
                        }
                    }
                } else {
                    addNewProblem(file);
                }
            }
        }
    }

    private void addNewProblem(File f) {
        if (f.isFile()) {
            Problem p = Settings.definition.createInputInstance();
            p.initializeProblem(f);
            p.loadProblem();
            if (!p.isValidInstance()) {
                return;
            }
            problems.add(p);
            problemNames.add(f.getName());
            editTimes.add(f.lastModified());
            problemSet.put(p.instanceName(), p);
            ProblemPane pane = new ProblemPane(p, this);
            visualizations.add(pane);
            add(pane);
        }
    }

    private void selectProblem(Problem p) {
        selectedProblem = p;
        for (ProblemListener l : listeners) {
            l.updateProblem(p);
        }
    }

    private void select(ProblemPane selected) {
        synchronized (problemLock) {
            for (ProblemPane pane : visualizations) {
                if (pane == selected) {
                    pane.selectComponent();
                    parent.getContentPanel().setView(pane.getProblem(), true);
                    selectProblem(pane.getProblem());
                } else {
                    pane.deselectComponent();
                }
            }
        }
        reloadGraphics();
    }

    public Problem getSelectedProblem() {
        return selectedProblem;
    }

    public void updateSelectedProblem(Class clazz) {
        synchronized (problemLock) {
            directory = null;
            checkProblemDirectory();
        }
        parent.getAlgorithmsPanel().loadAlgorithms();
        selectProblem(null);
        parent.getContentPanel().setView(null, true);
        parent.getSettingsPanel().problemClassUpdated();

    }

    public Problem getProblem(String name) {
        return problemSet.getOrDefault(name, null);
    }

    public static ArrayList<Problem> getProblems() {
        return problems;
    }

    public static void setProblems(ArrayList<Problem> p) {
        problems = p;
    }

    public static HashMap<String, Problem> getProblemSet() {
        return problemSet;
    }

    public static void setProblemSet(HashMap<String, Problem> problemSet) {
        ProblemsPane.problemSet = problemSet;
    }

    private class ProblemPane extends JPanel implements MouseListener {

        ProblemsPane parent;
        Problem problem;

        ProblemPane(Problem problem, ProblemsPane parent) {
            super();
            this.parent = parent;
            this.problem = problem;
            //layout
            GridBagLayout layout = new GridBagLayout();
            super.setLayout(layout);

            JLabel label = new JLabel(problem.instanceName());
            label.setMaximumSize(new Dimension(100, 30));
            setMaximumSize(new Dimension(300, 30));

            GridBagConstraints c = new GridBagConstraints();
            c.fill = GridBagConstraints.BOTH;
            c.weightx = 1;
            c.weighty = 1;
            c.gridwidth = 1;
            c.gridheight = 1;
            //c.insets = new Insets(10,10,10,10);
            c.gridx = 0;
            c.gridy = 0;
            super.add(label, c);

            setBorder(new BevelBorder(BevelBorder.RAISED));

            addMouseListener(this);
        }

        public void deselectComponent() {
            setBorder(new BevelBorder(BevelBorder.RAISED));
        }

        public void selectComponent() {
            setBorder(new BevelBorder(BevelBorder.LOWERED));
        }

        @Override
        public void mouseClicked(MouseEvent e) {

        }

        @Override
        public void mousePressed(MouseEvent e) {

        }

        @Override
        public void mouseReleased(MouseEvent e) {
            parent.select(this);
        }

        @Override
        public void mouseEntered(MouseEvent e) {

        }

        @Override
        public void mouseExited(MouseEvent e) {

        }

        public Problem getProblem() {
            return problem;
        }

        public void setProblem(Problem problem) {
            this.problem = problem;
        }
    }

    public void reloadGraphics() {
        revalidate();
        repaint();
    }

    public File getDirectory() {
        return directory;
    }

    public void setDirectory(File directory) {
        synchronized (problemLock) {
            this.directory = directory;
            if (parent != null) {
                parent.getContentPanel().setView(null, true);
            }
        }
    }
}
