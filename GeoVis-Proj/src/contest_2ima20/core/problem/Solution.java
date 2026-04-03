package contest_2ima20.core.problem;

import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;

public abstract class Solution implements Viewable, Serializable {

    /**
     * Duration before solution was constructed
     */
    private long duration;
    private String name;

    public final String getName() {
        return name;
    }

    public final void setName(String name) {
        this.name = name;
    }

    public long getDuration() {
        return duration;
    }

    public void setDuration(long duration) {
        this.duration = duration;
    }

    public abstract boolean isValid();

    public abstract double computeQuality();

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
}
