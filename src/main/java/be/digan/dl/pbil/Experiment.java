package be.digan.dl.pbil;

public class Experiment {
    final private double[] input;
    final private int output;

    public Experiment(double[] input, int output) {
        this.input = input;
        this.output = output;
    }

    public double[] getInput() {
        return input;
    }

    public int getOutput() {
        return output;
    }

}
