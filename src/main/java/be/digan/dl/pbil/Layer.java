package be.digan.dl.pbil;

public interface Layer {
    double[] calculate(final double[] wiehgts, final double[] input);

    int getWeightCount();

    int setWeightIndex(int index);
}
