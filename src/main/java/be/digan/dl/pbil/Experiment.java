package be.digan.dl.pbil;

public class Experiment {
    final private long[] input;
    final private int output;

    public Experiment(long[] input, int output) {
        this.input = input;
        this.output = output;
    }

    public long[] getInput() {
        return input;
    }

    public int getOutput() {
        return output;
    }

}
