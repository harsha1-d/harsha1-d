/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contest_2ima20.client;

import contest_2ima20.core.util.Settings;
import contest_2ima20.client.gui.MainPane;
import javax.swing.WindowConstants;

/**
 *
 * @author Wouter Meulemans (w.meulemans@tue.nl)
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Settings.init("guisettings.txt");
        MainPane frame = new MainPane();
        frame.setVisible(true);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

}
