/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contest_2ima20.client.gui;

import contest_2ima20.client.algorithms.AlgorithmList;
import contest_2ima20.core.problem.Viewable;
import contest_2ima20.core.util.Settings;
import contest_2ima20.client.problem.Algorithm;
import contest_2ima20.core.problem.Problem;
import contest_2ima20.core.problem.Solution;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import javax.swing.*;

/**
 *
 * @author Wout
 */
public class SettingsPane extends JPanel implements ActionListener {

    private JButton inputButton;
    private JButton solutionInputButton;
    private JButton advancedButton;
    private JButton writeView;
    private JButton resetZoomButton;
    private JToggleButton problemViewToggl;
    private JFileChooser solutionPicker;
    private JFileChooser problemPicker;
    private JFileChooser exportPicker;
    private GridBagLayout layout;
    private MainPane parent;

    private static File problemsPath;

    private static File solutionPath;

    private static File exportPath = null;
    private static boolean autoExportSolutions;

    public SettingsPane(MainPane parent) {
        super();

        String path = Settings.getValue("problemPath", null);
        if (path != null) {
            File file = new File(path);
            if (file.exists() && file.isDirectory()) {
                problemsPath = file;
                parent.getProblemsPanel().setDirectory(problemsPath);
            }
        }
        if (problemsPath == null) {
            problemsPath = new File(System.getProperty("user.home"));
        }

        path = Settings.getValue("solutionPath", null);
        if (path != null) {
            File file = new File(path);
            if (file.exists()) {
                solutionPath = file;
            }
        }
        if (solutionPath == null) {
            solutionPath = new File(System.getProperty("user.home"));
        }

        path = Settings.getValue("exportPath", null);
        if (path != null) {
            File file = new File(path);
            if (file.exists()) {
                exportPath = file;
            }
        }

        autoExportSolutions = Settings.getBoolean("autoExportSolutions", false);
        if (autoExportSolutions && exportPath == null) {
            autoExportSolutions = false;
        }

        super.setPreferredSize(new Dimension(100, 50));
        super.setBorder(BorderFactory.createMatteBorder(3, 0, 0, 0, Color.darkGray));
        this.parent = parent;

        //define Components
        inputButton = new JButton("Select input directory");
        solutionInputButton = new JButton("Select custom solution");
        writeView = new JButton("Write to file");
        problemViewToggl = new JToggleButton("View Problem");
        resetZoomButton = new JButton("Reset view (spacebar)");
        advancedButton = new JButton("Advanced");

        problemViewToggl.setSelected(true);
        inputButton.addActionListener(this);
        solutionInputButton.addActionListener(this);
        writeView.addActionListener(this);
        advancedButton.addActionListener(this);
        problemViewToggl.addActionListener(this);
        resetZoomButton.addActionListener(this);

        problemPicker = new JFileChooser();
        problemPicker.setFileFilter(new FolderFilter());
        problemPicker.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        if (problemsPath.exists()) {
            problemPicker.setCurrentDirectory(problemsPath);
        } else {
            problemPicker.setCurrentDirectory(new File(System.getProperty("user.home")));
        }

        solutionPicker = new JFileChooser();
        solutionPicker.setFileFilter(new FolderFilter());
        solutionPicker.setFileSelectionMode(JFileChooser.FILES_ONLY);

        solutionPicker = new JFileChooser();
        solutionPicker.setCurrentDirectory(solutionPath);

        exportPicker = new JFileChooser();
        exportPicker.setFileFilter(new FolderFilter());
        exportPicker.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        if (exportPath != null && exportPath.exists()) {
            exportPicker.setCurrentDirectory(exportPath);
        } else {
            exportPath = null;
        }

        //layout
        layout = new GridBagLayout();
        super.setLayout(layout);

        Component[] components = new Component[]{inputButton, solutionInputButton, resetZoomButton, writeView,
            advancedButton};

        int xCoord = 0;
        for (Component comp : components) {
            GridBagConstraints c = new GridBagConstraints();
            c.fill = GridBagConstraints.BOTH;
            c.weightx = 1;
            c.weighty = 1;
            c.gridwidth = 1;
            c.gridheight = 1;
            c.insets = new Insets(10, 10, 10, 10);
            c.gridx = xCoord;
            c.gridy = 0;
            xCoord++;
            super.add(comp, c);
        }
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == inputButton) {
            int returnVal = problemPicker.showOpenDialog(parent);

            if (returnVal == JFileChooser.APPROVE_OPTION) {
                if (problemPicker.getSelectedFile().isDirectory()) {
                    File file = problemPicker.getSelectedFile();
                    parent.getProblemsPanel().setDirectory(file);
                    Settings.setValue("problemPath", file.getAbsolutePath());
                    inputButton.setText("Select other problem directory");
                }
            }
        } else if (e.getSource() == solutionInputButton) {
            if (parent.getProblemsPanel().getSelectedProblem() == null) {
                JOptionPane.showMessageDialog(parent, "Please first select a problem.");
                return;
            }

            int returnVal = solutionPicker.showOpenDialog(parent);

            if (returnVal == JFileChooser.APPROVE_OPTION) {
                if (!solutionPicker.getSelectedFile().isDirectory()) {
                    File file = solutionPicker.getSelectedFile();
                    Solution solution = parent.getProblemsPanel().getSelectedProblem().loadSolution("Custom", file);
                    Settings.setValue("solutionPath", file.getAbsolutePath());
                    parent.getAlgorithmsPanel().syncViews();
                    if (solution == null || !solution.isValid()) {
                        JOptionPane.showMessageDialog(parent, "The provided solution is not valid for this problem.");
                    } else {
                        writeSolution(parent.getProblemsPanel().getSelectedProblem().instanceName() + "-custom", solution);
                        parent.getContentPanel().setView(solution, true);
                        parent.getAlgorithmsPanel().syncViews();
                    }
                }
            }
        } else if (e.getSource() == writeView) {
            Viewable v = parent.getContentPanel().getViewable();
            if (v instanceof Solution) {
                if (exportPath == null || !exportPath.exists()) {
                    int returnVal = exportPicker.showOpenDialog(parent);
                    if (returnVal == JFileChooser.APPROVE_OPTION) {
                        if (exportPicker.getSelectedFile().isDirectory()) {
                            exportPath = exportPicker.getSelectedFile();
                            Settings.setValue("exportPath", exportPath.getAbsolutePath());
                        }
                    } else {
                        return;
                    }
                }

                Solution s = (Solution) v;
                String name = parent.getProblemsPanel().getSelectedProblem().instanceName() + "-" + s.getName();
                if (writeSolution(name, s)) {
                    JOptionPane.showMessageDialog(parent, "Succesfully written to file");
                } else {
                    JOptionPane.showMessageDialog(parent, "There was a problem creating the file");
                }

            } else {
                JOptionPane.showMessageDialog(parent, "Only solutions can be written to a file");
            }
        } else if (e.getSource() == advancedButton) {
            JDialog dialog = new JDialog(parent);
            dialog.setContentPane(new AdvancedPane(parent, dialog));
            dialog.setSize(new Dimension(450, 375));
            dialog.setModal(true);
            dialog.pack();
            dialog.setResizable(false);
            dialog.setLocationRelativeTo(null);
            dialog.setVisible(true);
        } else if (e.getSource() == resetZoomButton) {
            parent.getContentPanel().zoomToFit();
        }
    }

    public static boolean writeSolution(String name, Solution s) {
        File exportFile = new File(exportPath.getAbsolutePath() + File.separator + name + ".txt");
        return s.write(exportFile);
    }

    public void batchReader(File directory) {
        System.out.println("Reading " + directory.getAbsolutePath());
        File[] files = directory.listFiles();
        for (File file : files) {
            if (Thread.interrupted()) {
                System.out.println("Thread was interrupted, file reading might have thrown an error");
                return;
            }
            String[] parts = file.getName().split("\\W");
            if (parts.length >= 3) {                                  //two parts plus extension
                // assume algorithms name is a single word
                String problemName = parts[0];
                for (int i = 1; i < parts.length - 2; i++) {
                    problemName += "-" + parts[i];
                }
                String solutionAlgo = parts[parts.length-2]; // before last, last is extension
                Problem p = parent.getProblemsPanel().getProblem(problemName);
                boolean correctAlgo = false;
                for (Algorithm algo : AlgorithmList.getAlgorithms()) {
                    if (solutionAlgo.equals(algo.className())) {
                        correctAlgo = true;
                        break;
                    }
                }
                if (solutionAlgo.equalsIgnoreCase("custom")) {
                    correctAlgo = true;
                }
                if (p != null && correctAlgo) {
                    Solution s = p.loadSolution(solutionAlgo, file);
                    System.out.println("Added " + file.getName());
                } else {
                    System.out.println("Skipped " + file.getName());
                }
            } else {
                System.out.println("Skipped " + file.getName());
            }
        }
    }

    public void problemClassUpdated() {
        inputButton.setText("Select input directory");
    }

    public static File getExportPath() {
        return exportPath;
    }

    public static void setExportPath(File exportPath) {
        SettingsPane.exportPath = exportPath;
        Settings.setValue("exportPath", exportPath.getAbsolutePath());
    }

    public static boolean isAutoExportSolutions() {
        return autoExportSolutions;
    }

    public static void setAutoExportSolutions(boolean solutions) {
        autoExportSolutions = solutions;
        Settings.setBoolean("autoExportSolutions", autoExportSolutions);
    }

    private static class FolderFilter extends javax.swing.filechooser.FileFilter {

        @Override
        public boolean accept(File file) {
            return file.isDirectory();
        }

        @Override
        public String getDescription() {
            return "Directories only";
        }
    }
}
