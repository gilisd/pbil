package be.digan.dl.pbil;

import java.util.Arrays;

public class Softmax implements Layer {


    @Override
    public double[] calculate(double[] weights, double[] input) {
        double[] exp = Arrays.stream(input).map(i -> Math.exp(i)).toArray();
        double sum = Arrays.stream(exp).sum();

        double[] collect = Arrays.stream(exp).map(i -> i / sum).toArray();
        return collect;
    }

    @Override
    public int getWeightCount() {
        return 0;
    }


    @Override
    public int setWeightIndex(int weightIndex) {
        return weightIndex;
    }
}
