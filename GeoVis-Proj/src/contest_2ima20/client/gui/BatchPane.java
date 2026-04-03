package contest_2ima20.client.gui;

import contest_2ima20.client.algorithms.AlgorithmList;
import contest_2ima20.client.problem.Algorithm;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

public class BatchPane extends JDialog implements ActionListener {

    ArrayList<ActionListener> listeners;
    ArrayList<AlgoPane> panes;
    JButton ok;
    JButton cancel;

    public BatchPane(Window owner, String title){
        super(owner, title);

        listeners = new ArrayList<>();
        panes = new ArrayList<>();

        JPanel content = new JPanel();
        content.setLayout(new BorderLayout());
        this.setContentPane(content);

        JPanel approvalPane = new JPanel();
        approvalPane.setLayout(new GridBagLayout());
        ok = new JButton("Solve");
        cancel = new JButton("Cancel");
        ok.addActionListener(this);
        cancel.addActionListener(this);

        GridBagConstraints c = new GridBagConstraints();
        c.fill = GridBagConstraints.BOTH;
        c.gridwidth = 1;
        c.gridheight = 1;
        c.gridx = 0;
        c.gridy = 0;
        c.weightx = 1;
        c.weighty = 1;

        approvalPane.add(cancel, c);
        c.gridx = 1;
        approvalPane.add(ok, c);
        content.add(approvalPane, BorderLayout.SOUTH);

        JPanel algoContainer = new JPanel();
        algoContainer.setLayout(new GridBagLayout());

        JScrollPane contentContainer = new JScrollPane(algoContainer);
        content.add(contentContainer, BorderLayout.CENTER);

        algoContainer.setLayout(new GridBagLayout());
        c = new GridBagConstraints();
        c.fill = GridBagConstraints.HORIZONTAL;
        c.gridwidth = 1;
        c.gridheight = 1;
        c.gridx = 0;
        c.gridy = 0;
        c.weightx = 1;
        c.weighty = 1;

        int yPos = 0;
        for(Algorithm algo : AlgorithmList.getAlgorithms()){
            AlgoPane p = new AlgoPane(algo.className());
            panes.add(p);
            c.gridy = yPos;
            algoContainer.add(p,c);
            yPos++;
        }

        setModal(true);
        setSize(new Dimension(250, 300));
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if(e.getSource() == ok){
            String command = "";
            for(AlgoPane p : panes){
                if(p.isChecked()){
                    if(!command.isEmpty()){
                        command += ",";
                    }
                    command += p.getName();
                }
            }
            for(ActionListener l :listeners){
                l.actionPerformed(new ActionEvent(this, 0, command));
            }
        }
        dispose();
    }

    public void addActionListener(ActionListener e){
        listeners.add(e);
    }

    public class AlgoPane extends JPanel {
        JLabel lblName;
        JCheckBox checked;

        public AlgoPane(String name){
            super();
            GridBagLayout l = new GridBagLayout();
            setLayout(l);

            checked = new JCheckBox();

            GridBagConstraints c = new GridBagConstraints();
            c.fill = GridBagConstraints.HORIZONTAL;
            c.gridx = 0;
            c.gridy = 0;
            c.gridwidth = 1;
            c.gridheight = 1;
            c.weightx = 1;
            c.weighty = 1;
            c.insets = new Insets(3,3,3,3);
            add(checked, c);

            lblName = new JLabel(name);
            c = new GridBagConstraints();
            c.fill = GridBagConstraints.HORIZONTAL;
            c.gridx = 1;
            c.gridy = 0;
            c.gridwidth = 3;
            c.gridheight = 1;
            c.weightx = 3;
            c.weighty = 1;
            c.insets = new Insets(3,3,3,3);
            add(lblName, c);
        }

        public boolean isChecked(){
            return checked.isSelected();
        }

        public String getName(){
            return lblName.getText();
        }
    }
}
