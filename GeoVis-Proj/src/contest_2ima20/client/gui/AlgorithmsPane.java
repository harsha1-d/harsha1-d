package contest_2ima20.client.gui;

import contest_2ima20.client.algorithms.AlgorithmList;
import contest_2ima20.core.problem.Viewable;
import contest_2ima20.client.problem.Algorithm;
import contest_2ima20.core.problem.Solution;
import contest_2ima20.core.problem.Problem;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.net.URL;
import java.util.ArrayList;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

public class AlgorithmsPane extends JPanel implements ProblemListener, ViewListener {

    private MainPane parent;
    private ArrayList<AlgorithmPane> algorithmsVisualizations;
    private AlgorithmPane customSolution;

    public AlgorithmsPane(MainPane parent) {
        super();
        this.parent = parent;
        parent.getProblemsPanel().addListener(this);
        parent.getContentPanel().addViewListener(this);

        algorithmsVisualizations = new ArrayList<>();

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        JPanel header = new JPanel();

        GridBagLayout layout = new GridBagLayout();
        header.setLayout(layout);

        header.setMaximumSize(new Dimension(300, 30));

        JLabel name = new JLabel("Algorithm name");
        name.setMaximumSize(new Dimension(140, 40));
        name.setMinimumSize(new Dimension(140, 40));
        name.setPreferredSize(new Dimension(140, 40));

        JLabel valid = new JLabel("Valid");
        valid.setMaximumSize(new Dimension(40, 40));
        valid.setMinimumSize(new Dimension(40, 40));
        valid.setPreferredSize(new Dimension(40, 40));

        JLabel penalty = new JLabel("Penalty");
        penalty.setMaximumSize(new Dimension(60, 40));
        penalty.setMinimumSize(new Dimension(60, 40));
        penalty.setPreferredSize(new Dimension(60, 40));

        GridBagConstraints c = new GridBagConstraints();
        c.fill = GridBagConstraints.BOTH;
        c.weightx = 1;
        c.weighty = 1;
        c.gridwidth = 3;
        c.gridheight = 1;
        c.insets = new Insets(0, 2, 0, 5);
        c.gridx = 0;
        c.gridy = 0;
        header.add(name, c);

        c = new GridBagConstraints();
        c.fill = GridBagConstraints.BOTH;
        c.insets = new Insets(0, 5, 0, 10);
        c.weightx = 1;
        c.weighty = 1;
        c.gridwidth = 1;
        c.gridheight = 1;
        c.gridx = 3;
        c.gridy = 0;
        header.add(valid, c);

        c = new GridBagConstraints();
        c.fill = GridBagConstraints.BOTH;
        c.insets = new Insets(0, 5, 0, 10);
        c.weightx = 1;
        c.weighty = 1;
        c.gridwidth = 1;
        c.gridheight = 1;
        c.gridx = 4;
        c.gridy = 0;
        header.add(penalty, c);

        add(header);
        loadAlgorithms();
        syncViews();
    }

    public void loadAlgorithms() {
        for (AlgorithmPane pane : algorithmsVisualizations) {
            remove(pane);
        }

        algorithmsVisualizations = new ArrayList<>();


        for (Algorithm algo : AlgorithmList.getAlgorithms()) {
            AlgorithmPane algoPane = new AlgorithmPane(algo, this);
            algorithmsVisualizations.add(algoPane);
            add(algoPane);
        }

        customSolution = new AlgorithmPane(null, this, "custom");
        customSolution.setVisible(false);
        algorithmsVisualizations.add(customSolution);
        add(customSolution);

        reloadGraphics();
    }

    public void syncViews() {
        if (parent.getProblemsPanel().getSelectedProblem() != null) {
            customSolution.setVisible(parent.getProblemsPanel().getSelectedProblem().getSolution("custom") != null);
        }
        for (AlgorithmPane pane : algorithmsVisualizations) {
            pane.syncView();
        }
    }

    public class AlgorithmPane extends JPanel implements MouseListener {

        AlgorithmsPane parent;
        Algorithm algo;
        String name;
        boolean selectable;
        JLabel imageLabel;
        JLabel validLabel;
        JLabel scoreLabel;
        Thread worker;
        Problem currentProblem;

        AlgorithmPane(Algorithm algo, AlgorithmsPane parent) {
            this(algo, parent, algo.getClass().getSimpleName());
        }

        AlgorithmPane(Algorithm algo, AlgorithmsPane parent, String name) {
            super();
            this.parent = parent;
            this.algo = algo;
            this.name = name;

            selectable = true;

            //layout
            GridBagLayout layout = new GridBagLayout();
            setLayout(layout);
            setMaximumSize(new Dimension(300, 40));

            JLabel label = new JLabel(name);
            label.setMaximumSize(new Dimension(100, 40));
            label.setMinimumSize(new Dimension(100, 40));
            label.setPreferredSize(new Dimension(100, 40));

            GridBagConstraints c = new GridBagConstraints();
            c.fill = GridBagConstraints.BOTH;
            c.weightx = 1;
            c.weighty = 1;
            c.gridwidth = 2;
            c.gridheight = 1;
            c.insets = new Insets(10, 3, 10, 0);
            c.gridx = 0;
            c.gridy = 0;
            add(label, c);

            URL source = getClass().getResource("/resources/play.png");
            ImageIcon icon = new ImageIcon(source);
            imageLabel = new JLabel(icon);
            imageLabel.setMinimumSize(new Dimension(40, 40));
            imageLabel.setPreferredSize(new Dimension(40, 40));
            imageLabel.setMaximumSize(new Dimension(40, 40));

            c = new GridBagConstraints();
            c.fill = GridBagConstraints.BOTH;
            c.insets = new Insets(0, 5, 0, 0);
            c.weightx = 1;
            c.weighty = 1;
            c.gridwidth = 1;
            c.gridheight = 1;
            c.gridx = 2;
            c.gridy = 0;

            add(imageLabel, c);

            validLabel = new JLabel("");
            validLabel.setHorizontalAlignment(SwingConstants.CENTER);
            validLabel.setMinimumSize(new Dimension(40, 40));
            validLabel.setPreferredSize(new Dimension(40, 40));
            validLabel.setMaximumSize(new Dimension(40, 40));

            c = new GridBagConstraints();
            c.fill = GridBagConstraints.BOTH;
            c.insets = new Insets(10, 5, 10, 10);
            c.weightx = 1;
            c.weighty = 1;
            c.gridwidth = 1;
            c.gridheight = 1;
            c.gridx = 3;
            c.gridy = 0;
            add(validLabel, c);

            scoreLabel = new JLabel("");
            scoreLabel.setHorizontalAlignment(SwingConstants.LEFT);
            scoreLabel.setMinimumSize(new Dimension(60, 40));
            scoreLabel.setPreferredSize(new Dimension(60, 40));
            scoreLabel.setMaximumSize(new Dimension(60, 40));

            c = new GridBagConstraints();
            c.fill = GridBagConstraints.BOTH;
            c.insets = new Insets(10, 10, 10, 10);
            c.weightx = 2;
            c.weighty = 1;
            c.gridwidth = 1;
            c.gridheight = 1;
            c.gridx = 4;
            c.gridy = 0;
            add(scoreLabel, c);

            imageLabel.addMouseListener(this);

            selectProblem(null);
        }

        public void selectProblem(Problem problem) {
            if (problem == currentProblem) {
                return;
            }
            if (worker != null) {
                worker.interrupt();
                worker = null;
            }
            currentProblem = problem;
            syncView();
        }

        public void syncView() {
            scoreLabel.setText("");
            validLabel.setText("");
            validLabel.setOpaque(false);

            if (worker != null && worker.getState() == Thread.State.TERMINATED) {
                worker = null;
            }

            if (currentProblem == null) {
                imageLabel.setIcon(null);
            } else {
                Solution solution = currentProblem.getSolution(name);
                if (solution == null && worker == null) {

                    imageLabel.setIcon(new ImageIcon(getClass().getResource("/resources/play.png")));
                    validLabel.setText("");
                    validLabel.setOpaque(false);

                } else if (solution == null) {

                    imageLabel.setIcon(new ImageIcon(getClass().getResource("/resources/hourglass.png")));
                    validLabel.setText("");
                    validLabel.setOpaque(false);

                } else {
                    double q = solution.computeQuality();
                    if (q < 0) {
                        scoreLabel.setText("");
                    } else {
                        scoreLabel.setText("" + ((int) (q * 100.0) / 100.0));
                    }

                    if (solution.isValid()) {
                        validLabel.setText("True");
                        validLabel.setOpaque(true);
                        validLabel.setBackground(new Color(134, 255, 134, 80));

                    } else {
                        validLabel.setText("False");
                        validLabel.setOpaque(true);
                        validLabel.setBackground(new Color(255, 51, 51, 80));
                    }

                    if (parent.parent.getContentPanel().getViewable().equals(solution)) {
                        imageLabel.setIcon(new ImageIcon(getClass().getResource("/resources/eye_visable.png")));
                    } else {
                        imageLabel.setIcon(new ImageIcon(getClass().getResource("/resources/eye_available.png")));
                    }
                }
            }
            reloadGraphics();
        }

        @Override
        public void mouseClicked(MouseEvent e) {

        }

        @Override
        public void mousePressed(MouseEvent e) {

        }

        @Override
        public void mouseReleased(MouseEvent e) {
            if (currentProblem != null) {
                Solution s = currentProblem.getSolution(name);
                if (s != null && parent.parent.getContentPanel().getViewable() != s) {
                    parent.parent.getContentPanel().setView(s, false);
                } else if (s != null && parent.parent.getContentPanel().getViewable() == s) {
                    parent.parent.getContentPanel().setView(currentProblem, false);
                } else if (s == null && worker == null) {
                    worker = algo.startAlgorithm(currentProblem, this);
                }
                parent.syncViews();
            }
        }

        @Override
        public void mouseEntered(MouseEvent e) {

        }

        @Override
        public void mouseExited(MouseEvent e) {

        }

        public Algorithm getAlgo() {
            return algo;
        }

        public void setAlgo(Algorithm algo) {
            this.algo = algo;
        }
    }

    public void reloadGraphics() {
        validate();
        doLayout();
        repaint();
    }

    @Override
    public void updateProblem(Problem p) {
        for (AlgorithmPane pane : algorithmsVisualizations) {
            pane.selectProblem(p);
        }
        syncViews();
    }

    @Override
    public void updateView(Viewable v) {
        syncViews();
    }

}
