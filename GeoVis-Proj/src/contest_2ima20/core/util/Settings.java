/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contest_2ima20.core.util;

import contest_2ima20.core.problem.ProblemDefinition;
import contest_2ima20.core.schematrees.SchematicTrees;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Properties;

/**
 *
 * @author Wouter Meulemans (w.meulemans@tue.nl)
 */
public class Settings {

    public static ProblemDefinition definition = new SchematicTrees();

    private static Properties props;
    private static String filename;

    public static void setDefaults(String... keyvalues) {
        for (int i = 0; i < keyvalues.length; i += 2) {
            if (!props.containsKey(keyvalues[i])) {
                props.setProperty(keyvalues[i], keyvalues[i + 1]);
            }
        }
        save();
    }

    public static String getValue(String key) {
        return getValue(key, null);
    }

    public static String getValue(String key, String deft) {
        return props.getProperty(key, deft);
    }

    public static void setValue(String key, String value) {
        props.setProperty(key, value);
        save();
    }

    public static int getInt(String key) {
        return getInt(key, 0);
    }

    public static int getInt(String key, int deft) {
        return Integer.parseInt(getValue(key, deft + ""));
    }

    public static void setInt(String key, int value) {
        setValue(key, value + "");
    }

    public static double getDouble(String key) {
        return getDouble(key, 0);
    }

    public static double getDouble(String key, double deft) {
        return Double.parseDouble(getValue(key, deft + ""));
    }

    public static void setDouble(String key, double value) {
        setValue(key, value + "");
    }

    public static boolean getBoolean(String key) {
        return getBoolean(key, false);
    }

    public static boolean getBoolean(String key, boolean b) {
        return Boolean.parseBoolean(getValue(key, "" + b));
    }

    public static void setBoolean(String key, boolean b) {
        setValue(key, b + "");
    }

    public static void init(String fname) {
        filename = fname;
        props = new Properties();

        try (InputStream input = new FileInputStream(filename)) {
            System.out.println("Reading settings from " + new File(filename).getAbsolutePath());
            props.load(input);
        } catch (IOException ex) {
            // Logger.getLogger(Settings.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void save() {
        try (OutputStream out = new FileOutputStream(filename)) {
            System.out.println("Storing settings to " + new File(filename).getAbsolutePath());
            props.store(out, "");
        } catch (IOException ex) {
            // Logger.getLogger(Settings.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void print() {
        List<Entry<String, String>> entries = new ArrayList(props.entrySet());

        entries.sort((e, f) -> String.CASE_INSENSITIVE_ORDER.compare(e.getKey(), f.getKey()));

        System.out.println("--------- SETTINGS ---------");
        for (Entry<String, String> e : entries) {
            System.out.println(e.getKey() + "=" + e.getValue());
        }
        System.out.println("----------------------------");

    }
}
