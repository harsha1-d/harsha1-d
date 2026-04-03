package contest_2ima20.client.problem;

import contest_2ima20.client.gui.AlgorithmsPane;
import contest_2ima20.client.gui.SettingsPane;
import contest_2ima20.client.networking.Sender;
import contest_2ima20.core.problem.Problem;
import contest_2ima20.core.problem.Solution;

public abstract class Algorithm<TInput extends Problem, TOutput extends Solution> {
    
    public final Thread startAlgorithm(TInput p, AlgorithmsPane.AlgorithmPane pane) {
        Thread runAlgo = new Thread(getRunnable(p, pane));
        runAlgo.start();
        return runAlgo;
    }

    public final Thread startAlgorithm(TInput p) {
        return startAlgorithm(p, null);
    }

    public final Runnable getRunnable(TInput input, AlgorithmsPane.AlgorithmPane pane) {
        return new Runnable() {
            @Override
            public void run() {
                System.out.println("Trying to solve problem " + input.instanceName() + " with algorithm " + className());
                try {
                    long time = System.currentTimeMillis();
                    TOutput output = null;
                    try {
                        output = doAlgorithm(input);
                    } catch (Exception e) {
                        System.err.println("Algorithm " + className() + " returned an error while calculating solution");
                        e.printStackTrace();
                    }

                    if (output == null) {
                        throw new NullPointerException("Calculated solution is null");
                    }

                    long duration = System.currentTimeMillis() - time;
                    output.setName(className());
                    output.setDuration(duration);
                    double seconds = duration / 1000.0;
                    input.putSolution(className(), output);
                    System.out.println("Solved problem " + input.instanceName()
                            + " with algorithm " + className()
                            + " in time " + seconds
                            + " and with score " + output.computeQuality()
                            + " " + (output.isValid() ? "(valid)" : "(INVALID)"));

                    if (SettingsPane.isAutoExportSolutions()) {
                        System.out.println("Writing solution " + input.instanceName() + "-" + output.getName() + " to output directory");
                        SettingsPane.writeSolution(input.instanceName() + "-" + output.getName(), output);
                    }
                    if (Sender.isJoinedContest()) {
                        System.out.println("Sending solution " + input.instanceName() + "-" + output.getName() + " to contest");
                        Sender.sendSolution(input, output);
                    }

                } catch (ClassCastException e) {
                    System.err.println("Template of problem did not return the correct solution type for algorithm class");
                    e.printStackTrace();
                } catch (NullPointerException e) {
                    e.printStackTrace();
                }
                if (pane != null) {
                    pane.syncView();
                }
            }
        };
    }

    public final Runnable getRunnable(TInput p) {
        return getRunnable(p, null);
    }

    public abstract TOutput doAlgorithm(TInput input);

    public String className() {
        return this.getClass().getSimpleName();
    }
}
