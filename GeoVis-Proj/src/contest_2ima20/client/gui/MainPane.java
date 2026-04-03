/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contest_2ima20.client.gui;


import java.awt.*;
import javax.swing.*;

/**
 *
 * @author Wout
 */
public class MainPane extends JFrame {

    private SettingsPane settingsPanel;
    private ContentPane contentPanel;
    private AlgorithmsPane algorithmsPanel;
    private ProblemsPane problemsPanel;

    public MainPane() {
        super("Algorithms for Geovisualization");

        //set size and location
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension d = t.getScreenSize();
        int width = (int) (d.width * 0.9);
        int height = (int) (d.height * 0.9);

        super.setPreferredSize(new Dimension(width, height));
        super.setSize(new Dimension(width, height));
        super.setLocation(d.width / 2 - this.getSize().width / 2, d.height / 2 - this.getSize().height / 2);

        // iconURL is null when not found
        ImageIcon icon = new ImageIcon(getClass().getResource("/resources/aga-logo.png"));
        setIconImage(icon.getImage());

        // layout of frame
        Container panel = super.getContentPane();
        BorderLayout layout = new BorderLayout();
        panel.setLayout(layout);

        contentPanel = new ContentPane();
        problemsPanel = new ProblemsPane(this, null);
        settingsPanel = new SettingsPane(this);
        algorithmsPanel = new AlgorithmsPane(this);


        ///// GENERATE SIDE PANE /////
        JPanel sidePane = new JPanel();
        sidePane.setPreferredSize(new Dimension(300, 100));
        sidePane.setBorder(BorderFactory.createMatteBorder(0, 3, 0, 0, Color.darkGray));

        GridBagLayout sideLayout = new GridBagLayout();
        sidePane.setLayout(sideLayout);

        JScrollPane upper = new JScrollPane(algorithmsPanel);
        upper.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        upper.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        upper.getVerticalScrollBar().setUnitIncrement(20);

        GridBagConstraints c = new GridBagConstraints();
        c.fill = GridBagConstraints.BOTH;
        c.weightx = 1;
        c.weighty = 0.5;
        c.gridwidth = 1;
        c.gridheight = 1;
        c.gridx = 0;
        c.gridy = 0;
        sidePane.add(upper, c);

        JScrollPane lower = new JScrollPane(problemsPanel);
        lower.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        lower.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        lower.getVerticalScrollBar().setUnitIncrement(20);

        c = new GridBagConstraints();
        c.fill = GridBagConstraints.BOTH;
        c.weightx = 1;
        c.weighty = 0.8;
        c.gridwidth = 1;
        c.gridheight = 1;
        c.gridx = 0;
        c.gridy = 1;
        sidePane.add(lower, c);

        /////// GENERATE CONTENT PANE ////////
        JPanel mainPane = new JPanel();

        GridBagLayout mainLayout = new GridBagLayout();
        mainPane.setLayout(mainLayout);

        c = new GridBagConstraints();
        c.fill = GridBagConstraints.BOTH;
        c.weightx = 0.5;
        c.weighty = 1;
        c.gridwidth = 1;
        c.gridheight = 1;
        c.gridx = 0;
        c.gridy = 0;
        mainPane.add(contentPanel, c);
  
        panel.add(mainPane, BorderLayout.CENTER);
        panel.add(settingsPanel, BorderLayout.SOUTH);
        panel.add(sidePane, BorderLayout.EAST);
    }

    public SettingsPane getSettingsPanel() {
        return settingsPanel;
    }

    public void setSettingsPanel(SettingsPane settingsPanel) {
        this.settingsPanel = settingsPanel;
    }

    public ContentPane getContentPanel() {
        return contentPanel;
    }

    public void setContentPanel(ContentPane contentPanel) {
        this.contentPanel = contentPanel;
    }

    public AlgorithmsPane getAlgorithmsPanel() {
        return algorithmsPanel;
    }

    public void setAlgorithmsPanel(AlgorithmsPane algorithmsPanel) {
        this.algorithmsPanel = algorithmsPanel;
    }

    public ProblemsPane getProblemsPanel() {
        return problemsPanel;
    }

    public void setProblemsPanel(ProblemsPane problemsPanel) {
        this.problemsPanel = problemsPanel;
    }

}
